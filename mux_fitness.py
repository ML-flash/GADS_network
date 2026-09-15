"""
mux_fitness.py

Multiplexer fitness for GADS, with the original MegaGP scoring and a rebalanced
one, so the two can be run head to head on the same seeds.

WHY REBALANCE

The original computes, per truth-table row:

    net_penalties = penalties + gene_penalties/1.4 - successful_operations

where gene_penalties is len(organism) and successful_operations credits 1.0 per
binary gate that fires and 0.6 per NOT. Per token that nets out as

    successful AND/OR/XOR   +1.000 - 0.714 = +0.286
    successful NOT          +0.600 - 0.714 = -0.114
    variable                 0.000 - 0.714 = -0.714
    underflow op            -1.000 - 0.714 = -1.714

Three consequences. A binary gate that fires is worth more than it costs, so
operator density pays. NOT is the only gate with negative net value, in a
problem whose primitive is a negated address line. And a junk segment of n
variables and n-1 gates costs 0.428n + 0.286 against 10 for a wrong row, so
roughly twenty junk tokens are cheaper than one row.

The reward was there to handle a real asymmetry: a gate whose second operand is
already on the stack needs one new token, one whose operands are both fresh
needs three. But the length term already charges exactly that, because it counts
tokens actually written. Crediting successful operations on top of it charges
the same structural fact twice and with the opposite sign.

So the rebalanced form drops the reward and adds the two things the original
cannot see:

    fitness = W_CORRECT * correct_rows
            - C_LEN   * len(organism)
            - C_FAIL  * underflow_events
            - C_RESID * max(0, final_stack_depth - 1)

Residue matters because evaluate_expression returns only the top of the stack.
Anything left underneath was never consumed and never checked, so under the
original scoring prefix junk is both nearly free and structurally invisible.

EVALUATION

Bitmask. Each variable is one integer whose bit r is that variable's value on
truth-table row r, so the whole table is evaluated in a single pass over the
organism and correctness is a popcount. This is what makes a 300 x 300 run
affordable; the row loop in the original is a 64x multiplier on every
evaluation.

ONE PLACE THIS DIVERGES FROM THE M-E-GA SOURCE, RECORDED RATHER THAN FIXED

The original interpreter reaches a binary gate with exactly one operand and does
`right = stack.pop()` before `left = stack.pop()` raises, so the one operand is
consumed and lost and the stack is left holding 0 in its place. _run below takes
the branch whole: with fewer than two operands it charges the failure and pushes
0 without popping, so that operand survives underneath. Depth after the event is
2 there and 1 in the original.

It changes nothing about which scoring is better and it is invisible to the
original scoring, which does not read depth. It is visible to C_RESID, which
charges the surviving operand as residue. megagp_machine.py reproduces the
original branch exactly and is the reference for what the M-E-GA interpreter
did; this file is the reference for what is being proposed instead.
"""

# ============================== PARAMETERS ==============================
# MEASURED, exp_mux_scoring.py, 6-multiplexer, 4 seeds, 400 generations. Mean
# best correct rows under the rebalanced scoring, sweeping C_LEN alone:
#
#     C_LEN   correct   selected len   dead   residue
#     0.10      47.5           7.5      0.0       0.0
#     0.25      46.5           8.2      0.0       0.0
#     0.50      50.0          13.8      0.0       0.0
#     1.00      44.5           3.0      0.0       0.0
#     1.50      42.0           2.5      0.0       0.0
#
# against 48.5 for the original scoring on the same seeds. So 1.5 is past the
# knee: at 0.15 rows per token the length term stops being parsimony and becomes
# the objective, the population collapses to two or three tokens, and the search
# never reaches a 27-token solution. At 0.50 the rebalance beats the original on
# correctness AND keeps dead and residue at zero, which the original cannot do at
# any coefficient because it does not have the terms.
#
# The two jobs separate cleanly here, and that is the point of the rebalance.
# C_RESID and C_FAIL buy the structural guarantee -- dead is 0.0 at every C_LEN
# in that table, including 0.10 -- and C_LEN alone trades search reach against
# program length. The original could not separate them: one coefficient carried
# both jobs, and carried the second with the wrong sign.
W_CORRECT = 10.0     # per correct truth-table row, both scorings
C_LEN     = 0.5      # per token, rebalanced only. See the sweep above.
C_FAIL    = 3.0      # per underflow, rebalanced only
C_RESID   = 3.0      # per unconsumed stack item beyond the answer

ORIG_DIV  = 1.4      # the original's length divisor
ORIG_NOT  = 0.6      # the original's credit for a successful NOT
ORIG_BIN  = 1.0      # the original's credit for a successful binary gate
# =======================================================================

OPS = ("NOT", "AND", "OR", "XOR")


class MuxFitness:
    """
    GADS fitness contract: .atoms is the alphabet, .score(atoms) returns a
    float. selection_lines S gives 2**S data lines and S + 2**S variables, so
    S=2 is the 6-multiplexer and S=3 the 11-multiplexer.
    """

    def __init__(self, selection_lines=2, scoring="rebalanced",
                 sample_rows=None):
        self.S = selection_lines
        self.D = 2 ** selection_lines
        self.n_var = self.S + self.D
        self.total_rows = 2 ** self.n_var
        if sample_rows is None:
            sample_rows = self.total_rows
        self.sample_rows = int(sample_rows)
        if self.sample_rows < 1:
            raise ValueError("sample_rows must be >= 1")
        if self.sample_rows > self.total_rows:
            raise ValueError("sample_rows cannot exceed the MUX truth table")
        self.n_rows = self.sample_rows
        self.scoring = scoring
        self.atoms = tuple("var%d" % i for i in range(self.n_var)) + OPS
        self._index = {a: i for i, a in enumerate(self.atoms)}
        self.FULL = (1 << self.n_rows) - 1
        self.row_ids = ()
        self.var_mask = []
        self.expected = 0
        if self.sample_rows == self.total_rows:
            self.set_rows(range(self.total_rows))

    def expected_for_row(self, row):
        """Return the MUX output for one integer-encoded input row."""
        if not 0 <= row < self.total_rows:
            raise ValueError("row outside MUX truth table")
        selected = row >> self.D
        return (row >> (self.D - selected - 1)) & 1

    def set_rows(self, row_ids):
        """
        Compile input/output mappings for one evaluation set into bitmasks.

        A row id is the integer encoding of (selection bits, data bits), with
        the first selection bit as the most significant bit. Only the selected
        rows are materialized, so memory and evaluation cost scale with the
        sample rather than the complete truth table.
        """
        rows = tuple(row_ids)
        if len(rows) != self.sample_rows:
            raise ValueError("expected %d sampled rows, got %d" %
                             (self.sample_rows, len(rows)))
        if len(set(rows)) != len(rows):
            raise ValueError("sampled rows must be unique")
        if any(row < 0 or row >= self.total_rows for row in rows):
            raise ValueError("sampled row outside MUX truth table")
        self.row_ids = rows
        self.var_mask = [0] * self.n_var
        self.expected = 0
        for sample_index, row in enumerate(rows):
            for variable in range(self.n_var):
                shift = self.n_var - variable - 1
                if (row >> shift) & 1:
                    self.var_mask[variable] |= 1 << sample_index
            if self.expected_for_row(row):
                self.expected |= 1 << sample_index

    def resample(self, rng):
        """
        Draw and compile the next generation's evaluation set.

        Sampling is without replacement. Exhaustive configurations keep the
        complete mapping and do not consume random state.
        """
        if self.sample_rows == self.total_rows:
            if not self.row_ids:
                self.set_rows(range(self.total_rows))
            return self.row_ids
        chosen = set()
        while len(chosen) < self.sample_rows:
            chosen.add(rng.randrange(self.total_rows))
        self.set_rows(sorted(chosen))
        return self.row_ids

    # ---- evaluation ----
    def _run(self, atoms):
        """
        Evaluate the whole truth table in one pass. Returns the result mask,
        the underflow count, the original's successful-operation credit, and
        the final stack depth.
        """
        if not self.row_ids:
            raise RuntimeError("no MUX evaluation rows are active; call resample()")
        st = []
        fails = 0
        succ = 0.0
        for a in atoms:
            if a[0] == "v":
                st.append(self.var_mask[int(a[3:])])
            elif a == "NOT":
                if st:
                    st.append((~st.pop()) & self.FULL); succ += ORIG_NOT
                else:
                    fails += 1; st.append(0)
            else:
                if len(st) >= 2:
                    r = st.pop(); l = st.pop()
                    st.append(l & r if a == "AND" else
                              (l | r if a == "OR" else l ^ r))
                    succ += ORIG_BIN
                else:
                    fails += 1; st.append(0)
        return (st[-1] if st else 0), fails, succ, len(st)

    def correct(self, atoms):
        res, _f, _s, _d = self._run(atoms)
        return bin(~(res ^ self.expected) & self.FULL).count("1")

    def score(self, atoms):
        res, fails, succ, depth = self._run(atoms)
        n_ok = bin(~(res ^ self.expected) & self.FULL).count("1")
        L = len(atoms)
        if self.scoring == "original":
            # penalties, length/1.4 and the operation credit are accumulated
            # per row and then averaged, so each is simply its own value.
            net = fails + L / ORIG_DIV - succ
            return W_CORRECT * n_ok - net
        return (W_CORRECT * n_ok
                - C_LEN * L
                - C_FAIL * fails
                - C_RESID * max(0, depth - 1))

    # ---- structure of a solution, for reporting ----
    def dead_tokens(self, atoms):
        """
        Count tokens that do not contribute to the returned value. Simulate the
        stack keeping the set of token positions behind each entry, then take
        everything outside the final entry's set.
        """
        st = []
        for i, a in enumerate(atoms):
            if a[0] == "v":
                st.append({i})
            elif a == "NOT":
                st.append((st.pop() | {i}) if st else {i})
            else:
                if len(st) >= 2:
                    r = st.pop(); l = st.pop(); st.append(l | r | {i})
                else:
                    st.append({i})
        live = st[-1] if st else set()
        return len(atoms) - len(live)

    def report(self, atoms):
        res, fails, succ, depth = self._run(atoms)
        n_ok = bin(~(res ^ self.expected) & self.FULL).count("1")
        return {"correct": n_ok, "rows": self.n_rows,
                "total_rows": self.total_rows, "len": len(atoms),
                "fails": fails, "residue": max(0, depth - 1),
                "dead": self.dead_tokens(atoms)}


CANONICAL_6MUX = ("var0 NOT var1 NOT AND var2 AND var0 NOT var1 AND var3 AND "
                  "var0 var1 NOT AND var4 AND var0 var1 AND var5 AND "
                  "OR OR OR").split()
