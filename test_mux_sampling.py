"""Regression checks for exhaustive and generation-sampled MUX evaluation."""
from __future__ import annotations

import random

from mux_fitness import CANONICAL_6MUX, MuxFitness


def bit_for_variable(row, n_var, variable):
    return (row >> (n_var - variable - 1)) & 1


def main():
    exhaustive = MuxFitness(selection_lines=2, scoring="rebalanced")
    report = exhaustive.report(CANONICAL_6MUX)
    assert report["correct"] == report["rows"] == report["total_rows"] == 64

    a = MuxFitness(selection_lines=4, scoring="rebalanced", sample_rows=257)
    b = MuxFitness(selection_lines=4, scoring="rebalanced", sample_rows=257)
    rng_a = random.Random(9127)
    rng_b = random.Random(9127)

    first = a.resample(rng_a)
    assert first == b.resample(rng_b)
    assert len(first) == len(set(first)) == 257
    assert a.n_var == 20
    assert a.total_rows == 2 ** 20

    for sample_index, row in enumerate(first):
        for variable in range(a.n_var):
            observed = (a.var_mask[variable] >> sample_index) & 1
            assert observed == bit_for_variable(row, a.n_var, variable)
        observed_output = (a.expected >> sample_index) & 1
        assert observed_output == a.expected_for_row(row)

    second = a.resample(rng_a)
    assert second != first
    assert second == b.resample(rng_b)

    # Large MUX construction is proportional to the sample, not 2**n_var.
    large = MuxFitness(selection_lines=5, scoring="original", sample_rows=128)
    large.resample(random.Random(3))
    assert large.n_var == 37
    assert large.total_rows == 2 ** 37
    assert len(large.row_ids) == 128

    print("MUX SAMPLING PASS")


if __name__ == "__main__":
    main()
