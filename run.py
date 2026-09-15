"""Reference implementation of an asynchronous GADS network.

The network semantics are:

- one OS process per node;
- private GADS state per node;
- captured compositions exchanged over multiprocessing queues;
- no barrier, lock, controller, consensus round, or shared generation clock;
- grid/torus nodes share each eligible batch to one randomly chosen neighbour;
- line/ring nodes share to all adjacent neighbours;
- arrival grace is the only network adapter;
- the master process is instrumentation only.

"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import random
import time
from pathlib import Path

import GADS as G
from mux_fitness import MuxFitness

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.json"
EVENT_KEYS = ("captures", "opens", "baseline_bounds", "open_bounds",
              "deletions", "used")
GADS_PARAMS = {
    "POPULATION_SIZE", "MIN_LEN", "MAX_LEN", "ENABLE_CROSSOVER",
    "NUM_PARENTS", "CROSSOVER_PROB", "BOUNDARY_INSERT_PROB",
    "BOUNDARY_REMOVE_PROB", "CAPTURE_PROB", "MIN_CAPTURE_LEN",
    "MUTATION_PROB", "BOUNDARY_MUTATION_PROB", "OPEN_PROB",
    "BASE_GENE_PROB", "MCO_DECAY", "DB_THRESHOLD", "ELITE_FRAC",
    "USE_FITNESS",
}


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def apply_gads_params(params: dict) -> None:
    unknown = sorted(set(params) - GADS_PARAMS)
    if unknown:
        raise ValueError("unknown GADS parameter(s): %s" % ", ".join(unknown))
    for name, value in params.items():
        if not hasattr(G, name):
            raise AttributeError("GADS.py has no parameter %s" % name)
        setattr(G, name, value)


def selection_lines_for_size(mux_size: int) -> int:
    mux_size = int(mux_size)
    for selection_lines in range(1, 32):
        candidate = selection_lines + 2 ** selection_lines
        if candidate == mux_size:
            return selection_lines
        if candidate > mux_size:
            break
    raise ValueError("MUX size must have the form S + 2**S (6, 11, 20, 37, ...)")


def bind_mux(mux_size: int, scoring: str, sample_rows=None) -> MuxFitness:
    selection_lines = selection_lines_for_size(mux_size)
    fit = MuxFitness(selection_lines=selection_lines, scoring=scoring,
                     sample_rows=sample_rows)
    G.FITNESS = fit
    G.ATOMS = fit.atoms
    G.N_BASE = len(fit.atoms)
    G.BASE_GENES = list(range(2, 2 + G.N_BASE))
    G.FIRST_COMP_ID = 2 + G.N_BASE
    return fit


def events():
    return {k: 0 for k in EVENT_KEYS}


def neighbours(i: int, n: int, topology: str) -> list[int]:
    """Return the bounded neighbours of one node."""
    if n <= 1:
        return []
    if topology == "ring":
        return sorted({(i - 1) % n, (i + 1) % n})
    if topology == "torus":
        side = int(round(math.sqrt(n)))
        if side * side != n:
            raise ValueError("torus needs a perfect-square node count, got %d" % n)
        r, c = divmod(i, side)
        adj = set()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            adj.add(((r + dr) % side) * side + ((c + dc) % side))
        adj.discard(i)
        return sorted(adj)
    if topology == "grid":
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
        r, c = divmod(i, cols)
        adj = []
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < rows and 0 <= cc < cols:
                j = rr * cols + cc
                if j < n:
                    adj.append(j)
        return adj
    if topology == "line":
        out = []
        if i - 1 >= 0:
            out.append(i - 1)
        if i + 1 < n:
            out.append(i + 1)
        return out
    raise ValueError("topology must be line, ring, grid, or torus")


def structure_key(tpl) -> str:
    canon = json.dumps(tpl, separators=(",", ":"))
    return hashlib.blake2b(canon.encode(), digest_size=8).hexdigest()


def compute_light_stats(pop, mg, evals):
    base = comp = 0
    freq = {}
    for org in pop:
        for tok in org:
            if G.is_boundary(tok):
                continue
            if G.is_base(tok):
                base += 1
            else:
                comp += 1
                freq[tok] = freq.get(tok, 0) + 1
    nd = base + comp
    comp_frac = comp / nd if nd else 0.0
    dom_frac = max(freq.values()) / nd if (freq and nd) else 0.0
    fits = [e[0] for e in evals]
    dls = [e[1] for e in evals]
    return (comp_frac, dom_frac, mg.size(),
            sum(fits) / len(fits), max(fits), min(fits),
            sum(dls) / len(dls), max(dls))


def _worker_params(cfg: dict) -> dict:
    net = cfg["network"]
    return {
        "gads": dict(cfg.get("gads") or {}),
        "mux": dict(cfg["mux"]),
        "generations": int(cfg["generations"]),
        "share_min_age": int(net.get("share_min_age", 2)),
        "arrival_grace": int(net.get("arrival_grace", 2)),
        "singularity_threshold": float(net.get("singularity_threshold", 0.95)),
        "telemetry_every": int(cfg.get("telemetry_every", 5)),
        "show_progress": bool(cfg.get("show_progress", True)),
    }


def node_worker(node_id, seed, p, share_mode, inbox, outboxes,
                result_q, progress_q, telemetry_q):
    """One asynchronous network node with private GADS state."""
    apply_gads_params(p["gads"])
    sample_rows = p["mux"].get("sample_rows")
    fit = bind_mux(int(p["mux"]["size"]), str(p["mux"]["scoring"]),
                   None if sample_rows is None else int(sample_rows))
    case_rng = random.Random(int(p["mux"]["sample_seed"]))

    gens = p["generations"]
    min_age = p["share_min_age"]
    arrival_grace = p["arrival_grace"]
    rng = random.Random(seed)
    mg = G.MetaGenome()
    pop = G.init_population(rng)
    ev = events()

    best = -1e18
    curve = []
    restart_gens = []
    cap_gen = {}
    absorbed = idem = shared_out = 0

    for gen in range(gens):
        # All organisms at this node see one common evaluation sample for this
        # generation. Every node uses the same sample seed, so local generation
        # g maps to the same cases without a shared generation clock.
        fit.resample(case_rng)
        evals, service = G.evaluate_population(pop, mg)
        _cf, df, mgsz, mf, xf, _nf, mdl, _xdl = compute_light_stats(pop, mg, evals)

        # Local recolonization acts on this node only.
        if df >= p["singularity_threshold"]:
            mg = G.MetaGenome()
            pop = G.init_population(rng)
            cap_gen = {}
            restart_gens.append(gen)
            evals, service = G.evaluate_population(pop, mg)
            _cf, df, mgsz, mf, xf, _nf, mdl, _xdl = compute_light_stats(pop, mg, evals)

        if p["telemetry_every"] > 0 and gen % p["telemetry_every"] == 0:
            telemetry_q.put((node_id, gen, set(structure_key(t) for t in mg.mco)))

        if xf > best:
            best = xf
        curve.append(best)
        if p["show_progress"]:
            progress_q.put((node_id, gen, xf, mf, best, mgsz, mdl))

        for cid in service:
            if mg.age_reset(cid):
                ev["used"] += 1

        ev["deletions"] += len(mg.sweep(service))

        pre = set(mg.mco)
        pop, n_elite = G.make_next_generation(pop, evals, rng)
        for org in pop[n_elite:]:
            G.mutate_org(org, mg, rng, ev)

        # The post-mutation difference is local capture only. Network arrivals
        # are admitted later while draining the inbox and never enter cap_gen.
        for tpl in mg.mco:
            if tpl not in pre:
                cap_gen[tpl] = gen
        for tpl in [t for t in cap_gen if not mg.is_active(t)]:
            del cap_gen[tpl]

        if outboxes:
            to_share = [
                tpl for tpl in mg.mco
                if tpl in cap_gen
                and (gen - cap_gen[tpl]) == min_age
                and mg.is_active(tpl)
            ]
            if to_share:
                payload = (node_id, to_share)
                if share_mode == "random_one":
                    rng.choice(outboxes).put(payload)
                else:
                    for ob in outboxes:
                        ob.put(payload)
                shared_out += len(to_share)

            while True:
                try:
                    _src, batch = inbox.get_nowait()
                except Exception:
                    break
                for tpl in batch:
                    if mg.is_active(tpl):
                        idem += 1
                    else:
                        mg.age_enter(tpl, grace=arrival_grace)
                        absorbed += 1

    result_q.put((node_id, {
        "curve": curve,
        "best": best,
        "mco": mg.size(),
        "absorbed": absorbed,
        "idem": idem,
        "shared_out": shared_out,
        "restarts": len(restart_gens),
        "restart_gens": restart_gens,
        "captures": ev["captures"],
        "deletions": ev["deletions"],
        "used": ev["used"],
        "seed": seed,
    }))


def run_network(cfg: dict) -> dict:
    """Run one asynchronous GADS network and return instrumentation only."""
    n_nodes = int(cfg["network"]["nodes"])
    topology = str(cfg["network"]["topology"]).lower()
    if n_nodes < 1:
        raise ValueError("network.nodes must be >= 1")
    # Resolve once in the master solely to validate topology. Workers never read N.
    for i in range(n_nodes):
        neighbours(i, n_nodes, topology)

    p = _worker_params(cfg)
    p["mux"]["sample_seed"] = int(
        p["mux"].get("sample_seed", cfg.get("seed", 42))
    )
    if p["generations"] < 1:
        raise ValueError("generations must be >= 1")
    if p["share_min_age"] < 0 or p["arrival_grace"] < 0:
        raise ValueError("sharing ages must be non-negative")

    share_mode = "random_one" if topology in ("grid", "torus") else "all"
    inboxes = [mp.Queue() for _ in range(n_nodes)]
    result_q = mp.Queue()
    progress_q = mp.Queue()
    telemetry_q = mp.Queue()
    base_seed = int(cfg.get("seed", 42))

    procs = []
    for i in range(n_nodes):
        outs = [inboxes[j] for j in neighbours(i, n_nodes, topology)]
        procs.append(mp.Process(
            target=node_worker,
            args=(i, base_seed + i, p, share_mode, inboxes[i], outs,
                  result_q, progress_q, telemetry_q),
            name="gads-node-%d" % i,
        ))

    started = time.time()
    for proc in procs:
        proc.start()

    state = {}
    results = {}
    node_keys = {i: set() for i in range(n_nodes)}
    heartbeat = float(cfg.get("heartbeat_seconds", 2.0))
    last_hb = started

    while len(results) < n_nodes:
        while True:
            try:
                nid, gen, gmax, gmean, nbest, mgsz, mdl = progress_q.get_nowait()
            except Exception:
                break
            state[nid] = (gen, gmax, gmean, nbest, mgsz, mdl)
        while True:
            try:
                nid, _gen, keys = telemetry_q.get_nowait()
            except Exception:
                break
            node_keys[nid] = keys
        while True:
            try:
                nid, res = result_q.get_nowait()
            except Exception:
                break
            results[nid] = res

        # Failure detection is instrumentation only; it cannot influence a live node.
        dead_without_result = [
            (i, proc.exitcode) for i, proc in enumerate(procs)
            if proc.exitcode is not None and i not in results and proc.exitcode != 0
        ]
        if dead_without_result:
            for proc in procs:
                if proc.is_alive():
                    proc.terminate()
            for proc in procs:
                proc.join()
            raise RuntimeError("network node exited without result: %r" % dead_without_result)

        if cfg.get("show_progress", True) and heartbeat > 0 and time.time() - last_hb >= heartbeat:
            union = set()
            for keys in node_keys.values():
                union |= keys
            net_best = max((v[3] for v in state.values()), default=float("-inf"))
            print("[t=%5.0fs] net best %s | structures unique=%d" %
                  (time.time() - started, net_best, len(union)), flush=True)
            last_hb = time.time()
        time.sleep(0.02)

    for proc in procs:
        proc.join()
    bad = [(i, p_.exitcode) for i, p_ in enumerate(procs) if p_.exitcode != 0]
    if bad:
        raise RuntimeError("network worker failure: %r" % bad)

    union = set()
    for keys in node_keys.values():
        union |= keys
    out = {
        "execution_model": "async_process_per_node_queue_network",
        "topology": topology,
        "nodes": n_nodes,
        "base_seed": base_seed,
        "node_seeds": [base_seed + i for i in range(n_nodes)],
        "share_mode": share_mode,
        "mux": {
            "size": int(p["mux"]["size"]),
            "scoring": str(p["mux"]["scoring"]),
            "sample_rows": p["mux"].get("sample_rows"),
            "sample_seed": int(p["mux"]["sample_seed"]),
            "total_rows": 1 << int(p["mux"]["size"]),
        },
        "elapsed_seconds": time.time() - started,
        "unique_structures_last_telemetry": len(union),
        "results": {str(i): results[i] for i in sorted(results)},
        "totals": {
            "captures": sum(r["captures"] for r in results.values()),
            "absorbed": sum(r["absorbed"] for r in results.values()),
            "idem": sum(r["idem"] for r in results.values()),
            "shared_out": sum(r["shared_out"] for r in results.values()),
            "restarts": sum(r["restarts"] for r in results.values()),
        },
    }
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--nodes", type=int)
    parser.add_argument("--generations", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--topology", choices=("line", "ring", "grid", "torus"))
    parser.add_argument("--mux-size", type=int)
    parser.add_argument("--sample-rows", type=int)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--scoring", choices=("original", "rebalanced"))
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.nodes is not None:
        cfg.setdefault("network", {})["nodes"] = args.nodes
    if args.generations is not None:
        cfg["generations"] = args.generations
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.topology is not None:
        cfg.setdefault("network", {})["topology"] = args.topology
    if args.mux_size is not None:
        cfg.setdefault("mux", {})["size"] = args.mux_size
    if args.sample_rows is not None:
        cfg.setdefault("mux", {})["sample_rows"] = args.sample_rows
    if args.sample_seed is not None:
        cfg.setdefault("mux", {})["sample_seed"] = args.sample_seed
    if args.scoring is not None:
        cfg.setdefault("mux", {})["scoring"] = args.scoring

    result = run_network(cfg)
    print(json.dumps({
        "execution_model": result["execution_model"],
        "topology": result["topology"],
        "nodes": result["nodes"],
        "node_seeds": result["node_seeds"],
        "totals": result["totals"],
        "elapsed_seconds": result["elapsed_seconds"],
    }, indent=2))

    save = None if args.no_save else cfg.get("save_result")
    if save:
        path = Path(save)
        if not path.is_absolute():
            path = HERE / path
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        print("saved", path)


if __name__ == "__main__":
    mp.freeze_support()
    main()
