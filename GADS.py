"""
GADS: Generative Adaptive Dynamical Systems.

A genetic algorithm whose alphabet is not fixed. The population searches over
base atoms plus a set of captured compositions called the Meta-Genome, and that
set is produced by the population itself. What a single mutation can reach
therefore depends on structure the search authored, and the coordinate system of
the search moves as the search proceeds.

ENCODING

A composition is its content, carried by value as an immutable tuple. Identity
is structural equality: two compositions are the same object when and only when
their content matches, at every depth. There is no identifier space for
compositions and no table mapping names to content.

Three properties follow from that choice rather than being implemented
separately. Nothing needs to resolve a reference, so there is no cache and no
admission policy. A composition is meaningful to any process that receives it
without a shared namespace, since it carries its own content. And a dying
composition can be folded into its parents by value, so death conserves
information instead of erasing it.

SUBSTRATE

Two delimiters, ENC_OPEN and ENC_CLOSE, plus the atoms supplied by the fitness
environment. Nothing else is given. Every composition is derived: insertion lays
down a matched delimiter pair, capture lifts the delimited span into a single
token, and that token can then be nested by the same route. All structure above
the atom is produced by boundary mechanics operating on the population.

PARTICIPATION

Every composition lives in one pool, and the census taken while decoding the
population is the whole test: a composition no organism used this generation is
removed. Nothing is counted and nothing is given a grace period. The census runs
before breeding and mutation, so what it establishes is that nothing held the
composition at the moment the population was scored, which is the evidence
deletion needs and the only evidence available locally.

A composition must therefore earn its own use. Nested compositions are invisible
to the census, since decode reports only what an organism holds at top level, so
a composition that has become part of a larger one is dissolved into it. Be
independently useful or be inlined into the context where you are useful.

One seam sits beside this rule without being part of it. A composition arriving
from another process is in nobody's hands, so a census that does not see it has
not established non-use, only that sampling was never given the chance to offer
it. age_enter takes a grace for that case and sweep honours it. Nothing in this
file passes one and no single node run reaches it; it is an adapter for
asynchronous parallel operation, and the number lives in run_network.

Meta-Genome size is not set anywhere. It self-sizes to the balance between
capture inflow and use-or-die pruning, which makes it a negative feedback loop:
a larger Meta-Genome spreads a roughly fixed service budget more thinly, so more
entries go unused.

DEATH AS INLINING

When a composition dies, every tracked structure referencing it is rewritten
with the dead child's contents spliced in place, and identical results are
merged. The decoded atomic content of a parent is unchanged by the death of its
child. Death changes granularity, not extension, so the representation can churn
continuously without losing information. The population is not touched;
organisms holding a dead structure continue to decode it correctly.

DECODE

Iterative over an explicit stack, so there is no depth limit and no recursion.
A composition's decoded atomic content is independent of how deeply it is
nested.
"""

import random

# ---- Population ----
POPULATION_SIZE = 300

# Bounds on initial organism length only. MAX_LEN is read solely by
# init_population and never caps a running organism. MIN_LEN additionally acts
# as the floor below which mutation will not delete. Length is otherwise free
# and is managed by parsimony pressure in the fitness environment.
MIN_LEN = 2
MAX_LEN = 100

# Recombination. Off by default: in GADS recombination is expressed through the
# Meta-Genome, since a captured composition carries structure between organisms
# whole. Crossover is available for comparison.
ENABLE_CROSSOVER = False
NUM_PARENTS      = 200
CROSSOVER_PROB   = 0.50

# Boundary mechanics. Insertion lays down a delimiter pair; removal takes one
# away. These set the rate at which candidate spans appear for capture.
BOUNDARY_INSERT_PROB = 0.0035
BOUNDARY_REMOVE_PROB = 0.0020

# Capture. A delimited span of at least MIN_CAPTURE_LEN tokens becomes a single
# composition token.
CAPTURE_PROB = 0.09
MIN_CAPTURE_LEN = 2

# Two mutation rates, one per context. MUTATION_PROB applies outside a
# delimited span, BOUNDARY_MUTATION_PROB inside one, and each is split evenly
# across the four operators a position offers: point, swap, insert, delete. An
# operator that cannot apply at a position contributes nothing and the bands
# below it shift down, which is how an atom at depth 0 skips open.
#
# A delimiter takes the rate of the span it bounds, so the swap that moves it
# is BOUNDARY_MUTATION_PROB / 4, the same width as every other operator in that
# ladder. Point, insert and delete cannot apply to a delimiter: the first two
# break the matched-pair invariant and the third is remove_pair_at's job, which
# carries its own rate above alongside insertion.
MUTATION_PROB          = 0.05
BOUNDARY_MUTATION_PROB = 0.02

# Open. Expands a composition back into its contents inside an organism.
OPEN_PROB = 0.005

# Sampling. BASE_GENE_PROB is the chance a new token is a base atom rather than
# a composition. MCO_DECAY weights the composition draw by capture recency, so
# lower values concentrate sampling on the newest captures.
BASE_GENE_PROB = 0.57
MCO_DECAY = 0.90

# Generations a rewrite stays claimable in the pending table.
#
# Compositions do not use this. Non-use at the census is proof of non-use and
# there is nothing to count, so no threshold governs their removal. It survives
# because a rewrite is claimed by an organism rather than observed in one, and
# one evaluation pass decodes every organism in the population, which makes a
# single generation the complete evidence there too.
DB_THRESHOLD = 1

ELITE_FRAC = 0.02
FITNESS_THRESHOLD = 1000
OVER_PENALTY = 2

USE_FITNESS = True

ENC_OPEN, ENC_CLOSE = 0, 1

# The default environment binding. A runner binds its own environment before
# anything is evaluated -- run_network.py assigns FITNESS, ATOMS, N_BASE,
# BASE_GENES and FIRST_COMP_ID from the problem it runs, which is what
# harness._bind_alphabet does -- so this only has to be importable.
#
# fitness.py is the HRR environment and is not part of this repo. When it is
# absent the substrate is left unsized and every binder still works; when it is
# present the binding is the same as it always was.
try:
    import fitness as _fitness_mod
except ImportError:
    _fitness_mod = None

if _fitness_mod is None:
    FITNESS = None
    ATOMS = ()
else:
    FITNESS = _fitness_mod.FitnessFunction(target=FITNESS_THRESHOLD,
                                           over_penalty=OVER_PENALTY) \
              if USE_FITNESS else None
    ATOMS = (FITNESS.atoms if FITNESS is not None
             else _fitness_mod.FitnessFunction.atoms)
N_BASE = len(ATOMS)
BASE_GENES = list(range(2, 2 + N_BASE))
FIRST_COMP_ID = 2 + N_BASE


class MetaGenome:
    """
    The working alphabet above the atoms.

    mco holds the active compositions, keyed by the composition itself since a
    composition is its content. Insertion order is capture order, which is what
    sampling reads, and the key is the membership test, so one structure serves
    both accesses.

    The value is the arrival grace: how many further sweeps this entry survives
    unheld. It is zero for every composition the machine admits on its own,
    because capture lands a composition in an organism and the next census sees
    it held. A nonzero grace is an adapter for asynchronous parallel operation.
    See age_enter.

    Removal is decided by the census rather than by anything stored here.

    MG is not required to decode. decode() expands a composition from its own
    content and never consults this object's state, so an organism holding a
    composition that has been removed still decodes to the same atoms. What MG
    governs is the proposal distribution, through sample_composition, and
    whether open expands a composition or leaves it whole, through is_active.
    A near-empty MG therefore does not stall the machine; it removes the brake
    on expansion.

    A third structure, pending, carries rewrites back to the population.
    Inlining rewrites entries here and cannot touch organisms, since reaching
    into the population to edit tokens is the global controller the foundation
    derivation excludes. So an organism goes on holding a form this object has
    moved past. It still decodes to the same atoms and loses no fitness, but it
    expresses that content through a token no longer live, so it emits no usage
    signal and sampling cannot offer it.

    pending closes that without a search. When inlining rewrites old into new
    the pair is recorded here, and each organism collects what applies to it on
    the decode it was already performing.

    This is local in the sense the foundation derivation requires. The test is
    not whether population state is written, since mutation writes it every
    generation, but whether anything is used that the local site could not have
    had. Nothing is scanned, no model of the population is consulted, and no
    assumption about the problem is made. Decode already takes an encoding and
    returns a value; it now sometimes returns two, the decoded content and the
    current structural identity of what was presented. The organism receives
    only an answer about the token it supplied, on a channel that was already
    carrying data in both directions. It is a return value, not a push.

    Entries here are the one thing that still counts generations, because a
    rewrite is claimed by an organism rather than observed in one. A composition
    is judged by a census that sees the whole population at once; a rewrite is
    judged by whether anyone came to collect it. One evaluation pass decodes
    every organism, so a single generation is a complete pass over everybody who
    could hold the superseded form, and DB_THRESHOLD of one is that pass. A
    claim resets the count anyway, since claiming is decoding and an update
    still in service should not expire under an organism that has not reached
    it.

    Size is emergent. Nothing here bounds mco, and no constant sets a target
    size.
    """

    def __init__(self):
        self.mco = {}            # active composition -> grace left, in
                                 # capture order
        self.pending = {}        # superseded form -> [current form, age]

    # ---- Decode ----
    def decode(self, org):
        """
        Expand an organism to its atomic content.

        Iterative over an explicit stack, so nesting depth is unbounded and no
        recursion limit applies. Returns the atom sequence and the set of
        top-level compositions encountered, which is what marks them as used
        for the current generation.
        """
        top = set()
        result = []

        for tok in org:
            if isinstance(tok, tuple):
                top.add(tok)
                stack = [tok]
                while stack:
                    curr = stack.pop()
                    if isinstance(curr, tuple):
                        # Reverse order preserves left-to-right decoding.
                        for t in reversed(curr):
                            stack.append(t)
                    elif curr in (ENC_OPEN, ENC_CLOSE):
                        continue
                    else:
                        result.append(curr)
            elif tok in (ENC_OPEN, ENC_CLOSE):
                continue
            else:
                result.append(tok)

        return result, top

    # ---- Participation ----
    def age_enter(self, tpl, grace=0):
        """
        Admit a composition to the pool.

        Admission carries no grace in the machine, and capture never passes one.
        A captured composition lands directly in an organism, so the next census
        sees it held, and non-use at that census is a real reading.

        grace is an adapter for asynchronous parallel operation and is not part
        of the theory. Nothing in this file passes a nonzero value and nothing
        in a single node run reaches it. It exists because a structure arriving
        from another process lands in nobody's hands, so a census that does not
        see it has not established non-use, only that sampling was never given
        the chance to offer it. That is not the same fact, and deleting on it
        would mean no structure could ever cross between nodes.

        A grace of n makes the entry survive its next n sweeps unheld. It buys
        reachability, not tenure: once the grace runs out the ordinary rule
        applies with nothing carried over, and an arrival nobody took is gone.
        See run_network.ARRIVAL_GRACE, which is where the number lives.

        A composition not already held enters at the end, which is the position
        sample_composition weights most. One already held keeps the position it
        has and only its grace is rewritten.
        """
        self.mco[tpl] = grace

    def age_reset(self, tpl):
        """
        Mark a composition as used this generation. Returns whether it was
        present.

        The stored value is vestigial and nothing reads it. Use is now proved by
        membership of the census that sweep is handed, not by a counter, and
        this survives as the presence check that keeps a stale signal from
        writing into db.
        """
        if tpl in self.mco:
            self.mco[tpl] = 0
            return True
        return False

    def sweep(self, used):
        """
        Remove every composition the population did not use this generation.

        Non-use at the census is the whole test, and there is nothing to count.
        The generations-without-use threshold belonged to the named encoding,
        where displacement put an entry in the pool and it needed a generation
        to be reclaimed under its name. With a composition carrying its own
        content there is no name to reclaim and no cache to miss.

        Called immediately after the census and before the population changes,
        so a composition removed here is one no organism was holding. That is
        the ordering the old tick did not have: it ran after mutation had
        handed compositions out, and decided death from a census taken before
        the handout.

        Structure captured later in the generation is not a candidate here. It
        faces its first census next generation, and by then it is sitting in the
        organism capture put it in.

        Rewrites this produces are recorded in pending and claimed on the next
        decode, so removal here never strands a holder.

        The grace branch below is an adapter for asynchronous parallel
        operation and is not part of the rule. It is dead in every single node
        run, since nothing in this file admits a composition with a grace. See
        age_enter.
        """
        self.pending_tick()
        removed = []
        for tpl in [t for t in self.mco if t not in used]:
            # Adapter, not theory. An arrival from another process is in
            # nobody's hands, so this census has not established non-use, only
            # that sampling had no chance to offer it. Hold it and spend one.
            if self.mco.get(tpl, 0) > 0:
                self.mco[tpl] -= 1
                continue
            # A death rewrites every structure referencing it, so a candidate
            # later in this pass may no longer exist under its own key. Its
            # rewritten form carries forward and faces the next census.
            if tpl in self.mco:
                self._denature(tpl)
                removed.append(tpl)
        return removed

    # ---- Death as inlining ----
    def _denature(self, dead):
        """
        Remove a dead composition and splice its contents into every structure
        that referenced it.

        Because a composition is its content, a parent holding the dead child
        can have that child replaced by the child's own elements, and the
        parent's decoded atoms are unchanged. Information is conserved and only
        granularity moves. Structures that become identical after the rewrite
        are merged, keeping the younger age.
        """
        # A pending rewrite whose target is the composition about to die points
        # at a form that will not exist. The dead composition has no
        # replacement, so there is nothing to repoint to and the entry is
        # dropped. An organism still holding that superseded form simply keeps
        # it, which is the behaviour it would have had with no rewrite table at
        # all. Without this, claim_rewrites would hand out a token the
        # Meta-Genome no longer holds, breaking the invariant that every target
        # here is live.
        for k in [k for k, v in self.pending.items() if v[0] == dead]:
            del self.pending[k]

        def deep_degrade(tpl):
            if not isinstance(tpl, tuple):
                return tpl

            changed = False
            out = []
            for el in tpl:
                if el == dead:
                    out.extend(dead)
                    changed = True
                elif isinstance(el, tuple):
                    new_el = deep_degrade(el)
                    out.append(new_el)
                    if new_el != el:
                        changed = True
                else:
                    out.append(el)

            return tuple(out) if changed else tpl

        # Rebuild: drop the dead entry, rewrite the rest, and merge any
        # duplicates the rewrite produced. A merged entry keeps the position of
        # its first occurrence and the larger grace, which is the one with reach
        # left.
        new_mco = {}
        for t, grace in self.mco.items():
            if t == dead:
                continue
            d = deep_degrade(t)
            if d != t:
                self._record_rewrite(t, d)
            new_mco[d] = max(new_mco[d], grace) if d in new_mco else grace
        self.mco = new_mco

    # ---- Carrying rewrites back to the population ----
    def _record_rewrite(self, old, new):
        """
        Note that old has been superseded by new, and keep every chain flat.

        Composition happens on insert rather than on lookup. Two deaths in one
        tick can rewrite t into d and then d into e, and an organism holding t
        needs e, not d. Repointing every entry whose target was old keeps the
        invariant that every target in this table is a form the Meta-Genome
        currently holds, so a claim is one step and always lands on the live
        form. Walking a chain at lookup time cannot promise that, because an
        intermediate entry may have already aged out, which would leave the
        organism on a form that no longer exists.

        The dead composition is never recorded here. It has no replacement; it
        was spliced into its referencers and is simply gone. Entries that
        pointed at it are dropped at the head of _denature, which is what keeps
        the every-target-is-live invariant true across deaths as well as
        rewrites.
        """
        if old == new:
            return
        for ent in self.pending.values():
            if ent[0] == old:
                ent[0] = new
        self.pending[old] = [new, 0]

    def claim_rewrites(self, org):
        """
        Return org with any superseded top-level token replaced by its current
        form, or None if nothing applies.

        Only top-level tokens are consulted. A rewrite is recorded against a
        whole composition, and a nested change is already folded into the
        parent's own entry, so there is nothing below the top level to look up.

        A claim resets the entry's age for the same reason usage resets a
        composition's: another organism may hold the same superseded form and
        not have decoded yet, and expiring while still in service is exactly
        what the threshold exists to prevent.
        """
        if not self.pending:
            return None
        out = []
        changed = False
        for tok in org:
            if isinstance(tok, tuple) and tok in self.pending:
                ent = self.pending[tok]
                out.append(ent[0])
                ent[1] = 0
                changed = True
            else:
                out.append(tok)
        return out if changed else None

    def pending_tick(self):
        """
        Age the rewrite table and drop what nobody claimed.

        Same clock and same threshold as the age pool. An entry that survives
        DB_THRESHOLD generations without a claim has had a full independent
        evaluation cycle in which no organism decoded its key, which is proof
        of non-service rather than an inference from it.
        """
        for k in list(self.pending.keys()):
            self.pending[k][1] += 1
            if self.pending[k][1] >= DB_THRESHOLD:
                del self.pending[k]

    # ---- Capture ----
    def try_capture(self, content_tuple):
        """
        Admit a delimited span as a composition.

        Returns the composition and whether it is new. An already-active span
        returns itself unchanged, since structural identity makes a duplicate
        capture the same object rather than a second copy.
        """
        if len(content_tuple) < MIN_CAPTURE_LEN:
            return None, False
        if self.is_active(content_tuple):
            return content_tuple, False
        ent = self.pending.get(content_tuple)
        if ent is not None:
            # The span matches a form the Meta-Genome has moved past. Dedup is
            # a content match, and the current form is the same content, so
            # hand that back rather than admitting the superseded key beside
            # its own replacement.
            return ent[0], False
        self.age_enter(content_tuple)
        return content_tuple, True

    def is_active(self, tpl):
        return tpl in self.mco

    def open_resolve(self, el):
        """
        Expand a composition one level for an organism.

        An active composition is returned whole. An inactive one is a structure
        that died while an organism still held it, so it is expanded into its
        parts, which decodes to the same atoms.
        """
        if not isinstance(el, tuple):
            return [el]
        if self.is_active(el):
            return [el]
        out = []
        for sub in el:
            out.extend(self.open_resolve(sub))
        return out

    def size(self): return len(self.mco)
    def all_ids(self): return list(self.mco)

    def sample_composition(self, rng):
        """
        Draw a composition weighted by capture recency under MCO_DECAY. Lower
        decay concentrates the draw on recent captures; higher decay spreads it
        across the whole set.
        """
        n = len(self.mco)
        if n == 0: return None
        keys = list(self.mco)
        weights = [MCO_DECAY ** (n - j - 1) for j in range(n)]
        total = sum(weights); roll = rng.random() * total; cum = 0.0
        for j in range(n):
            cum += weights[j]
            if roll <= cum: return keys[j]
        return keys[-1]


def sample_token(rng, mg):
    """Draw a base atom or a composition. This is the proposal distribution,
    and the Meta-Genome is what parameterizes it."""
    if mg.size() == 0 or rng.random() < BASE_GENE_PROB:
        return rng.choice(BASE_GENES)
    c = mg.sample_composition(rng)
    return c if c is not None else rng.choice(BASE_GENES)

def init_population(rng):
    """Random organisms of base atoms only. No boundaries, no compositions,
    empty Meta-Genome. Everything above the atom is derived from here."""
    return [[rng.choice(BASE_GENES) for _ in range(rng.randint(MIN_LEN, MAX_LEN))]
            for _ in range(POPULATION_SIZE)]

def is_boundary(t): return t == ENC_OPEN or t == ENC_CLOSE
def is_base(t):     return isinstance(t, int) and 2 <= t < FIRST_COMP_ID
def is_comp(t):     return isinstance(t, tuple)

def assert_invariants(org):
    """Delimiters must be matched and never nested, and every plain token must
    be a known atom."""
    inside = False
    for tok in org:
        if tok == ENC_OPEN:
            if inside: raise AssertionError("Nesting")
            inside = True
        elif tok == ENC_CLOSE:
            if not inside: raise AssertionError("Unmatched close")
            inside = False
        elif isinstance(tok, tuple):
            continue
        elif tok < 2:
            raise AssertionError("Unknown")
    if inside: raise AssertionError("Unmatched open")

def remove_pair_at(org, idx):
    """Remove a delimiter and its partner together, so the pair invariant
    holds. Returns the index to resume from."""
    if org[idx] == ENC_OPEN:
        j = idx + 1
        while j < len(org) and org[j] != ENC_CLOSE: j += 1
        if j >= len(org): raise AssertionError("open rm")
        del org[j]; del org[idx]; return max(idx - 1, 0)
    if org[idx] == ENC_CLOSE:
        j = idx - 1
        while j >= 0 and org[j] != ENC_OPEN: j -= 1
        if j < 0: raise AssertionError("close rm")
        del org[idx]; del org[j]; return max(j - 1, 0)
    raise AssertionError("non-delim")

def find_boundaries(org, index):
    """Locate the delimiter pair enclosing a position."""
    s = None
    for i in range(index, -1, -1):
        if org[i] == ENC_OPEN: s = i; break
    if s is None: return None, None
    e = None
    for j in range(s + 1, len(org)):
        if org[j] == ENC_CLOSE: e = j; break
    return s, e

def calculate_depth(org, index):
    """Boundary depth at a position. Zero means outside any delimited span."""
    d = 0
    for c in org[:index + 1]:
        if c == ENC_OPEN: d += 1
        elif c == ENC_CLOSE: d -= 1
    return d

def attempt_capture(org, interior_idx, mg, events):
    """
    Lift the delimited span containing a position into a single composition
    token, replacing the span and its delimiters in the organism.
    """
    s, e = find_boundaries(org, interior_idx)
    if s is None or e is None: return False
    content = tuple(org[s + 1:e])
    tpl, is_new = mg.try_capture(content)
    if tpl is None: return False
    org[s:e + 1] = [tpl]
    if is_new: events["captures"] += 1
    return True

def can_swap(a, b): return not (is_boundary(a) and is_boundary(b))

def attempt_swap(org, i, rng):
    """Swap a token with a neighbour. Two delimiters never swap with each
    other, which would reorder a pair."""
    if len(org) < 2: return i, False
    dirs = [1, -1] if rng.random() < 0.5 else [-1, 1]
    for d in dirs:
        j = i + d
        if 0 <= j < len(org) and can_swap(org[i], org[j]):
            org[i], org[j] = org[j], org[i]; return j, True
    return i, False

def mutate_org(org, mg, rng, events):
    """
    Walk an organism and apply one operator per position.

    Two regimes. Outside a delimited span the operators are boundary insertion,
    open, substitution, swap, insertion and deletion. Inside one, insertion is
    replaced by capture, so a span that survives long enough becomes a
    composition. This is the only route by which new structure enters the
    Meta-Genome.

    A delimiter is taken before either regime, since depth does not describe it:
    it can be removed with its partner or swapped past a neighbour, and nothing
    else applies. Both operators draw at the rate of the span it bounds.
    """
    i = 0
    while i < len(org):
        tok = org[i]
        depth = calculate_depth(org, i); roll = rng.random()

        if is_boundary(tok):
            if roll < BOUNDARY_REMOVE_PROB:
                if len(org) - 2 >= MIN_LEN: i = remove_pair_at(org, i); continue
                else: i += 1; continue
            elif roll < BOUNDARY_REMOVE_PROB + BOUNDARY_MUTATION_PROB / 4:
                ni, ds = attempt_swap(org, i, rng)
                if ds: i = ni
                i += 1; continue
            else:
                i += 1; continue

        if depth == 0:
            ow = OPEN_PROB if is_comp(tok) else 0.0
            t0 = BOUNDARY_INSERT_PROB; t1 = t0 + ow; t2 = t1 + MUTATION_PROB / 4
            t3 = t2 + MUTATION_PROB / 4; t4 = t3 + MUTATION_PROB / 4; t5 = t4 + MUTATION_PROB / 4
            if roll < t0:
                org.insert(i, ENC_OPEN); org.insert(i + 2, ENC_CLOSE); i += 1; events["baseline_bounds"] += 1
            elif roll < t1:
                content = []
                for el in tok:
                    content.extend(mg.open_resolve(el))
                exp = [ENC_OPEN] + content + [ENC_CLOSE]
                org[i:i + 1] = exp; i += len(exp); events["opens"] += 1; events["open_bounds"] += 1
            elif roll < t2:
                org[i] = sample_token(rng, mg); i += 1
            elif roll < t3:
                ni, ds = attempt_swap(org, i, rng)
                if ds: i = ni
                i += 1
            elif roll < t4:
                nt = sample_token(rng, mg)
                if rng.random() < 0.5: org.insert(i, nt); i += 1
                else: org.insert(i + 1, nt); i += 2
            elif roll < t5:
                if len(org) > MIN_LEN: del org[i]; i = max(i - 1, 0)
                else: i += 1
            else:
                i += 1
            continue

        # Inside a delimited span.
        ow = OPEN_PROB
        t0 = CAPTURE_PROB; t1 = t0 + ow; t2 = t1 + BOUNDARY_MUTATION_PROB / 4
        t3 = t2 + BOUNDARY_MUTATION_PROB / 4; t4 = t3 + BOUNDARY_MUTATION_PROB / 4; t5 = t4 + BOUNDARY_MUTATION_PROB / 4
        if roll < t0:
            if attempt_capture(org, i, mg, events): i += 1; continue
            i += 1
        elif roll < t1:
            if is_comp(tok):
                content = []
                for el in tok:
                    content.extend(mg.open_resolve(el))
                org[i:i + 1] = content; i += len(content); events["opens"] += 1
            else:
                i += 1
        elif roll < t2:
            org[i] = sample_token(rng, mg); i += 1
        elif roll < t3:
            ni, ds = attempt_swap(org, i, rng)
            if ds: i = ni
            i += 1
        elif roll < t4:
            nt = sample_token(rng, mg)
            if rng.random() < 0.5: org.insert(i, nt); i += 1
            else: org.insert(i + 1, nt); i += 2
        elif roll < t5:
            if len(org) > MIN_LEN: del org[i]; i = max(i - 1, 0)
            else: i += 1
        else:
            i += 1
    assert_invariants(org)


# ---- Fitness and selection ----
def evaluate_fitness(org, mg):
    """Score an organism on its decoded atomic content. Fitness never sees
    compositions, only the atoms they expand to."""
    decoded, top = mg.decode(org)
    L = len(decoded)
    if FITNESS is None:
        fit = 0.0
    else:
        atoms = [ATOMS[t - 2] for t in decoded]
        fit = FITNESS.score(atoms)
    return fit, L, top

def evaluate_population(pop, mg):
    """
    Score the population and collect every composition used this generation,
    which is the set whose ages are reset.

    Each organism first collects any rewrite that applies to it. This runs
    before scoring rather than after, because the token an organism carries is
    what enters the service set: holding a superseded form emits a usage signal
    for a composition the Meta-Genome no longer has, so the live form ages as
    though nobody were using it. Claiming first puts the signal back on the
    entry that exists. Decoded content is identical either way, so no score
    moves because of it.
    """
    evals = []; service_set = set()
    for o in pop:
        updated = mg.claim_rewrites(o)
        if updated is not None:
            o[:] = updated
        fit, L, top = evaluate_fitness(o, mg)
        evals.append((fit, L, top))
        service_set |= top
    return evals, service_set

def safe_cut_points(org):
    """
    Positions where a cut leaves a balanced prefix, meaning every index at
    which boundary depth is zero. A cut here never lands inside a delimiter
    pair, so a balanced prefix joined to a balanced suffix is a valid organism.
    """
    pts = [0]; depth = 0
    for i, tok in enumerate(org):
        if tok == ENC_OPEN: depth += 1
        elif tok == ENC_CLOSE: depth -= 1
        if depth == 0: pts.append(i + 1)
    return pts

def crossover(a, b, rng):
    """
    Single-point splice at boundary-safe cuts on each parent. No length clamp;
    the child is whatever the splice produces. A composition is one token, so
    cuts fall on either side of it but never inside, and captured structure
    crosses over whole. The only floor is MIN_LEN.
    """
    ka = rng.choice(safe_cut_points(a))
    kb = rng.choice(safe_cut_points(b))
    child = a[:ka] + b[kb:]
    if len(child) < MIN_LEN:
        return a.copy()
    return child

def breed(parents, rng):
    """Produce one child, by splice when crossover is enabled and by copy
    otherwise. The disabled path draws no extra randomness."""
    if ENABLE_CROSSOVER and len(parents) >= 2 and rng.random() < CROSSOVER_PROB:
        return crossover(rng.choice(parents), rng.choice(parents), rng)
    return rng.choice(parents).copy()

def make_next_generation(pop, evals, rng):
    """
    Build the next population by rank selection.

    Returns the population and the number of elites, which are copied unchanged
    and are therefore not mutated by the caller. ELITE_FRAC of zero yields no
    elites, so every organism is mutated and contributes to capture.
    """
    n = len(pop)
    if not USE_FITNESS:
        n_par = min(NUM_PARENTS, n)
        parents = [rng.choice(pop).copy() for _ in range(n_par)]
        next_pop = [breed(parents, rng) for _ in range(POPULATION_SIZE)]
        return next_pop, 0

    ranked = sorted(range(n), key=lambda i: evals[i][0], reverse=True)
    n_elite = int(ELITE_FRAC * POPULATION_SIZE)
    n_par = min(NUM_PARENTS, len(ranked))
    elite = [pop[ranked[i]].copy() for i in range(min(n_elite, len(ranked)))]
    parents = [pop[ranked[i]].copy() for i in range(n_par)]
    next_pop = list(elite)
    while len(next_pop) < POPULATION_SIZE:
        next_pop.append(breed(parents, rng))
    return next_pop, n_elite

