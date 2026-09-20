"""Chapter 19: repeated trials, pass@k, pass^k and where the variation comes from.

pass@k and pass^k are checked against their definition, an average over every choice of
k of the recorded trials. The mean squares are checked against scipy's one-way ANOVA in
`scripts/check_reliability_against_libraries.py`.
"""

from itertools import combinations
from math import comb

import pytest

from agent_evals import reliability


def _brute(outcomes, k):
    subsets = list(combinations(outcomes, k))
    at = sum(any(s) for s in subsets) / len(subsets)
    hat = sum(all(s) for s in subsets) / len(subsets)
    return at, hat


@pytest.mark.parametrize("n", [3, 5, 8])
def test_pass_at_and_pass_hat_equal_the_average_over_every_subset_of_trials(n):
    for c in range(n + 1):
        outcomes = [True] * c + [False] * (n - c)
        for k in range(1, n + 1):
            at, hat = _brute(outcomes, k)
            assert reliability.pass_at(c, n, k) == pytest.approx(at)
            assert reliability.pass_hat(c, n, k) == pytest.approx(hat)


def test_at_k_equals_1_both_are_the_success_rate():
    assert reliability.pass_at(2, 3, 1) == pytest.approx(2 / 3)
    assert reliability.pass_hat(2, 3, 1) == pytest.approx(2 / 3)


def test_pass_hat_matches_the_tau_bench_formula():
    assert reliability.pass_hat(5, 8, 3) == pytest.approx(comb(5, 3) / comb(8, 3))


def test_a_case_that_never_succeeds_stays_at_zero_and_one_that_always_does_at_one():
    assert reliability.pass_at(0, 5, 3) == 0.0
    assert reliability.pass_hat(5, 5, 5) == 1.0
    assert reliability.pass_at(1, 5, 5) == 1.0


def test_pass_hat_is_not_the_success_rate_to_the_power_k_when_cases_differ():
    outs = {"easy": [True] * 4, "hard": [False] * 4}
    rows = reliability.curve(outs, range(1, 5))
    assert rows[0][2] == pytest.approx(0.5)
    assert rows[3][2] == pytest.approx(0.5)
    assert 0.5**4 < rows[3][2]


def test_the_split_counts_always_never_and_sometimes():
    outs = {
        "a": [True, True, True],
        "b": [False, False, False],
        "c": [True, False, True],
        "d": [True, True, True],
    }
    assert reliability.split_counts(outs) == (2, 1, 1)


def test_variance_components_on_a_hand_worked_example():
    # two cases, two trials: one always succeeds, one always fails.
    outs = {"a": [True, True], "b": [False, False]}
    msb, msw, var_between, var_within = reliability.variance_components(outs)
    assert msb == pytest.approx(1.0)
    assert msw == pytest.approx(0.0)
    assert var_between == pytest.approx(0.5)
    assert var_within == pytest.approx(0.0)


def test_variance_components_refuse_uneven_or_single_trials():
    with pytest.raises(ValueError):
        reliability.variance_components({"a": [True], "b": [False]})
    with pytest.raises(ValueError):
        reliability.variance_components({"a": [True, True], "b": [False]})


def test_more_trials_shrink_the_error_only_down_to_the_between_case_floor():
    floor = reliability.standard_error(0.18, 0.06, 42, 10**6)
    assert reliability.standard_error(0.18, 0.06, 42, 1) > floor
    assert reliability.standard_error(0.18, 0.06, 42, 8) > floor
    assert reliability.standard_error(0.18, 0.06, 84, 1) < reliability.standard_error(
        0.18, 0.06, 42, 8
    )


def test_a_half_and_half_case_looks_consistent_a_quarter_of_the_time_in_three_trials():
    assert reliability.chance_of_looking_consistent(0.5, 3) == pytest.approx(0.25)
    assert reliability.chance_of_looking_consistent(0.5, 8) == pytest.approx(2 / 256)
