"""Behavioral smoke for bounded asynchronous GADS exchange."""
from __future__ import annotations

from pathlib import Path

import run


def main():
    cfg = run.load_config(Path(__file__).with_name("config.smoke.json"))
    out = run.run_network(cfg)

    assert out["execution_model"] == "async_process_per_node_queue_network"
    assert out["topology"] == "grid"
    assert out["nodes"] == 4
    assert out["node_seeds"] == [42, 43, 44, 45]
    assert out["share_mode"] == "random_one"
    assert out["mux"]["size"] == 20
    assert out["mux"]["sample_rows"] == 128
    assert out["mux"]["total_rows"] == 2 ** 20
    assert out["totals"]["captures"] > 0
    assert out["totals"]["shared_out"] > 0
    assert out["totals"]["absorbed"] + out["totals"]["idem"] > 0

    source = Path(run.__file__).read_text(encoding="utf-8")
    for required in (
        "mp.Process(",
        "mp.Queue()",
        "inbox.get_nowait()",
        'share_mode = "random_one" if topology in ("grid", "torus") else "all"',
        "base_seed + i",
        "mg.age_enter(tpl, grace=arrival_grace)",
        "fit.resample(case_rng)",
    ):
        assert required in source
    for forbidden in ("Barrier(", "Pipe("):
        assert forbidden not in source

    print("NETWORK EXCHANGE PASS")
    print(" captures:", out["totals"]["captures"])
    print(" sends:", out["totals"]["shared_out"])
    print(" receives:", out["totals"]["absorbed"] + out["totals"]["idem"])


if __name__ == "__main__":
    main()
