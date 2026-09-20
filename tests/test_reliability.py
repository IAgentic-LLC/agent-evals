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


# ---------------------------------------------------------------- recorded runs


def _recorded():
    from pathlib import Path

    from agent_evals.runner import load_cases, read_traces

    root = Path(__file__).resolve().parents[1]
    cases = {
        c.case_id: c for c in load_cases(root / "datasets/triage_heldout_v1.jsonl")
    }
    runs = [
        read_traces(root / "runs" / n / "traces.jsonl") for n in reliability.TRIAL_RUNS
    ]
    return cases, runs


def _flat(text):
    return " ".join(text.split())


def test_the_recorded_runs_are_42_cases_by_8_trials():
    cases, runs = _recorded()
    outs = reliability.outcomes(cases, runs)
    assert len(outs) == 42 and {len(v) for v in outs.values()} == {8}


def test_the_first_three_runs_are_chapter_7s_26_25_and_28():
    cases, runs = _recorded()
    text = _flat(reliability.render_trials(cases, list(reliability.TRIAL_RUNS), runs))
    for line in ("v1 26 of 42", "v1-2 25 of 42", "v1-3 28 of 42"):
        assert line in text


def test_pass_at_flattens_and_pass_hat_falls_much_slower_than_independence_says():
    cases, runs = _recorded()
    text = _flat(reliability.render_curve(reliability.outcomes(cases, runs)))
    assert "1 64% 64% 51-77% 64%" in text
    assert "3 71% 57% 43-70% 27%" in text
    assert "8 71% 52% 38-67% 3%" in text


def test_cases_split_into_22_always_12_never_and_8_sometimes():
    cases, runs = _recorded()
    text = _flat(reliability.render_split(reliability.outcomes(cases, runs)))
    assert "succeeded every time 22 never succeeded 12 sometimes 8" in text


def test_most_of_the_spread_is_between_cases_not_between_runs():
    cases, runs = _recorded()
    text = _flat(reliability.render_variance(reliability.outcomes(cases, runs)))
    assert "share of the spread that is between cases: 80%" in text
    assert "the same 336 runs as 336 different cases: 2.6" in text


def test_ten_cases_end_in_an_error_every_time_and_the_errors_are_loops():
    _, runs = _recorded()
    errs = reliability.render_errors(reliability.errors_by_case(runs))
    assert "8 of 8 10" in _flat(errs) and "0 of 8 26" in _flat(errs)
    kinds = _flat(reliability.render_error_kinds(runs))
    assert "ToolLoopDidNotConverge 85 HandoffLoopDetected 15 runs in all 336" in kinds


def test_three_trials_mostly_classify_a_case_the_way_eight_do():
    cases, runs = _recorded()
    text = _flat(reliability.render_classes(reliability.outcomes(cases, runs)))
    assert "always 22 0 1" in text and "never 0 12 0" in text
    assert "sometimes 0 0 7" in text


def test_the_reliability_reports_fit_the_page(capsys):
    from agent_evals import cli

    for part in ("trials", "curve", "split", "variance", "errors", "classes"):
        assert cli.main(["reliability", "--part", part]) == 0
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line


def test_the_errors_sit_on_the_technical_tickets():
    cases, runs = _recorded()
    outs = reliability.outcomes(cases, runs)
    errs = reliability.errors_by_case(runs)
    text = _flat(reliability.render_by_specialist(cases, outs, errs))
    assert "billing 13 12 0 1 2" in text
    assert "security 12 8 2 2 0" in text
    assert "technical 17 2 10 5 98" in text


def test_the_unrequired_freeze_repeats_on_nine_tickets_and_no_forbidden_action_occurs():
    cases, runs = _recorded()
    text = _flat(reliability.render_actions(cases, runs, "freeze_account"))
    assert "8 of 8 9" in text and "0 of 8 32" in text and "2 of 8 1" in text
    assert "runs that took a forbidden action: 0 of 336" in text
