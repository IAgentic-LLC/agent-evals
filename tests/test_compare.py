"""Chapter 18: is one run really different from another.

The reference values below were computed once with scipy and statsmodels and are pinned
here, so this code is checked against established implementations.
"""

import random
from pathlib import Path

import pytest

from agent_evals import abstention, cli, compare
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]


def _has(text, *fragments):
    flat = " ".join(text.split())
    for fragment in fragments:
        assert " ".join(fragment.split()) in flat, fragment


# ------------------------------------------------------------------ the tests


def test_exact_sign_test_matches_scipy_binomtest():
    assert compare.exact_sign_test(1, 4) == pytest.approx(0.375)
    assert compare.exact_sign_test(4, 6) == pytest.approx(0.75390625)
    assert compare.exact_sign_test(9, 1) == pytest.approx(0.021484375)
    assert compare.exact_sign_test(0, 5) == pytest.approx(0.0625)
    assert compare.exact_sign_test(0, 0) == 1.0


def test_paired_counts_sum_to_the_number_of_questions():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    assert compare.paired_counts(a, b) == (1, 2, 1, 1)


def test_holm_and_benjamini_hochberg_match_statsmodels():
    p = [0.01, 0.04, 0.03, 0.005, 0.2]
    assert compare.holm(p) == pytest.approx([0.04, 0.09, 0.09, 0.025, 0.2])
    assert compare.benjamini_hochberg(p) == pytest.approx(
        [0.025, 0.05, 0.05, 0.025, 0.2]
    )


def test_newcombe_matches_statsmodels():
    low, high = compare.newcombe_difference(26, 110, 28, 110)
    assert low == pytest.approx(-0.1309832764176716, abs=1e-9)
    assert high == pytest.approx(0.09519386737591189, abs=1e-9)


# --------------------------------------------------------------- the intervals


def test_the_cluster_interval_is_reproducible_and_holds_the_estimate():
    values = [0.0, 0.5, 1.0, 0.0, 0.0, 0.5, 0.0, 0.0]
    once = compare.cluster_interval(compare.mean, values, resamples=800)
    assert once == compare.cluster_interval(compare.mean, values, resamples=800)
    assert once[0] <= compare.mean(values) <= once[1]


def test_wald_undercovers_a_rare_rate_at_small_n_and_wilson_does_not():
    wald = compare.coverage(compare.wald_interval, 55, 0.03)
    wilson = compare.coverage(compare.wilson_interval, 55, 0.03)
    assert wald == pytest.approx(0.81, abs=0.01)
    assert wilson > 0.95


def test_coverage_of_a_hand_worked_case():
    def only_middle(k, n):
        return (0.0, 1.0) if k == 1 else (2.0, 3.0)

    assert compare.coverage(only_middle, 2, 0.5) == pytest.approx(0.5)


def test_bayes_interval_is_symmetric_when_counts_are_and_moves_with_the_data():
    low, high = compare.bayes_paired_interval(5, 5, 20, 20, draws=3000)
    assert low == pytest.approx(-high, abs=0.02)
    low, high = compare.bayes_paired_interval(20, 2, 10, 20, draws=3000)
    assert low > 0


# ---------------------------------------------------------------------- power


def test_power_is_the_false_alarm_rate_when_there_is_no_gap():
    assert compare.power_sign_test(100, 0.0, 0.2) < 0.05


def test_exact_power_agrees_with_a_simulation():
    rng = random.Random(1)
    n, gap, d = 60, 0.10, 0.25
    cells = [(d + gap) / 2, (d - gap) / 2, 1 - d]
    hits = 0
    for _ in range(3000):
        kinds = rng.choices(range(3), weights=cells, k=n)
        hits += compare.exact_sign_test(kinds.count(0), kinds.count(1)) < 0.05
    assert compare.power_sign_test(n, gap, d) == pytest.approx(hits / 3000, abs=0.03)


def test_power_rises_with_questions_and_with_the_gap():
    assert compare.power_sign_test(200, 0.1, 0.2) > compare.power_sign_test(
        100, 0.1, 0.2
    )
    assert compare.power_sign_test(100, 0.15, 0.2) > compare.power_sign_test(
        100, 0.1, 0.2
    )
    assert compare.questions_needed(0.10, 0.19) == 160


def test_a_gap_larger_than_the_discordance_is_refused():
    with pytest.raises(ValueError):
        compare.power_sign_test(50, 0.3, 0.1)


# --------------------------------------------------------------- recorded runs


def _inputs():
    cases = {c.case_id: c for c in load_cases(ROOT / "datasets/pkg_abstain_v1.jsonl")}
    runs = {
        name: [read_traces(ROOT / "runs" / r / "traces.jsonl") for r in dirs]
        for name, dirs in abstention.CONFIGS.items()
    }
    return cases, runs


def test_the_same_questions_should_be_answered_in_every_run():
    cases, runs = _inputs()
    outcomes = compare.question_outcomes(cases, runs)
    for per_question in outcomes.values():
        assert len(per_question) == 55
        assert all(len(v) == 2 for v in per_question.values())


def test_the_intervals_reproduce_chapter_14s_counts():
    text = compare.render_intervals(*_inputs())
    _has(text, "shipped prompt 28/110 17-34% 18-34% 15-36%")
    _has(text, "permissive prompt 26/110 16-32% 17-32% 14-35%")


def test_pairing_narrows_the_interval_on_the_difference():
    text = compare.render_difference(*_inputs())
    _has(
        text,
        "as two unrelated groups of answers -1.8 -13.1 to 9.5",
        "same questions, redrawn together -1.8 -6.4 to 3.6",
        "by answer (110 pairs) 6 4 0.75",
        "by question (55) 4 1 0.38",
    )


def test_the_same_prompt_run_twice_differs_about_as_much_as_two_prompts():
    _has(
        compare.render_noise(*_inputs()),
        "shipped, run 1 against run 2 2 2 1.00",
        "permissive, run 1 against run 2 3 1 0.62",
        "shipped against permissive, run 1 3 3 1.00",
        "shipped against permissive, run 2 3 1 0.62",
    )


def test_many_slices_raise_the_chance_of_a_false_alarm_and_holm_removes_it():
    text = compare.render_slices(*_inputs())
    _has(text, "1 1% 1% 1%", "20 16% 0% 0%")


def test_the_plan_needs_hundreds_of_questions_for_a_small_gap():
    text = compare.render_plan(*_inputs())
    _has(
        text,
        "5 of 55",
        "55 8% 29% 55%",
        "for 80% 466 161 88",
    )


def test_the_attack_is_not_a_close_call():
    text = compare.attack_report(ROOT)
    _has(
        text,
        "prompt v2 43 90 38-58%",
        "prompt v3 0 90 0-4%",
        "p = 2.3e-13",
        "questions (33): v2 missed more on 23, v3 on 0, tied 10",
        "question-level p = 2.4e-07",
        "0 of 33 (0-10%)",
    )


def test_paired_coverage_runs_and_pairing_makes_the_interval_narrower():
    got = compare.paired_interval_coverage(55, 0.1, 0.09, 0.2, studies=40)
    assert got["paired bootstrap"][1] < got["unpaired"][1]
    assert got["paired Bayes"][1] < got["unpaired"][1]


def test_the_report_commands_fit_the_page(capsys):
    for part in ("intervals", "difference", "noise", "plan", "coverage", "attack"):
        assert cli.main(["compare", "--part", part]) == 0
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line
