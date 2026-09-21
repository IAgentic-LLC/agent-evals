"""Chapter 23: how the gate's rules behave when nothing changed and when a drop is real.

The enumeration is checked on small synthetic passes where the right answer is known.
The tables are pinned against the recorded runs of chapter 22, every split tried and none
sampled, so they are the same every time.
"""

from functools import cache
from pathlib import Path

from agent_evals import cli, gate_study, regression
from agent_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "datasets/triage_heldout_v1.jsonl"


def _flat(text):
    return " ".join(text.split())


def _pass(met: int, total: int = 40):
    return {f"T{i}": i < met for i in range(total)}


@cache
def _recorded():
    cases = {c.case_id: c for c in load_cases(HELD)}
    return gate_study.load(ROOT, cases)


@cache
def _nothing_changed():
    return gate_study.pooled(
        [gate_study.nothing_changed(v) for v in _recorded().values()]
    )


def test_a_pass_is_one_trial_of_every_ticket_taken_from_each_run():
    data = _recorded()
    assert set(data) == set(gate_study.CONDITIONS)
    assert all(len(p) == 6 and len(p[0]) == 42 for p in data.values())


def test_the_measured_drop_is_positive_when_the_candidate_is_worse():
    good = [_pass(40)] * 6
    bad = [_pass(30)] * 6
    assert gate_study.measured_drop(good, bad) == 25.0
    assert gate_study.measured_drop(bad, good) == -25.0


def test_six_identical_passes_split_into_threes_are_never_told_apart():
    outcomes = gate_study.nothing_changed([_pass(30)] * 6)
    assert all(o.block == 0 and o.hold == 0 for o in outcomes.values())


def test_a_drop_on_every_ticket_is_blocked_by_every_rule_in_every_draw():
    outcomes = gate_study.real_drop([_pass(40)] * 6, [_pass(0)] * 6)
    assert all(o.block == 1 for o in outcomes.values())


def test_an_outcome_reports_the_rest_as_passed():
    assert gate_study.Outcome(0.25, 0.5).passed == 0.25


def test_the_pooled_outcome_is_the_mean_over_conditions():
    a = {label: gate_study.Outcome(0.0, 0.2) for label, _, _ in gate_study.RULES}
    b = {label: gate_study.Outcome(0.2, 0.0) for label, _, _ in gate_study.RULES}
    pooled = gate_study.pooled([a, b])
    assert all(
        abs(o.block - 0.1) < 1e-9 and abs(o.hold - 0.1) < 1e-9 for o in pooled.values()
    )


def test_when_nothing_changed_the_point_rule_blocks_a_few_and_the_interval_rules_hold_some():
    pooled = _nothing_changed()
    rows = {
        k.split(",")[0]: (round(100 * v.block), round(100 * v.hold))
        for k, v in pooled.items()
    }
    assert rows["point"] == (4, 0)
    assert rows["sign test"] == (1, 0)
    assert pooled["interval, margin 10"].block == 0
    assert round(100 * pooled["interval, margin 10"].hold) == 10
    assert round(100 * pooled["interval, margin 5"].hold) == 41


def test_the_drops_table_is_pinned_and_a_gain_is_never_blocked():
    text = _flat(_render("drops"))
    assert "3.6 + guard -> 3.6-flash +22.6 100/0 100/0 75/25 100/0" in text
    assert "2.5-flash -> 3.6-flash +9.9 96/0 1/0 0/100 0/100" in text
    assert "3.6-flash -> 3.6 + guard -22.6 0/0 0/0 0/0 0/0" in text


def test_the_retry_table_is_pinned_and_a_retry_costs_little_detection_for_the_point_rule():
    text = _flat(_render("retry"))
    assert "point, drop over 5 2 / 0 96 / 93 100 / 100" in text
    assert "interval, margin 5 41 / 25 100 / 100 100 / 100" in text


def test_the_simulation_is_seeded_so_it_gives_the_same_answer_twice():
    data = _recorded()
    a = gate_study.retry_nothing_changed(data["3.6-flash"], 40, seed=3)
    b = gate_study.retry_nothing_changed(data["3.6-flash"], 40, seed=3)
    assert a == b


def test_what_one_candidate_costs_in_model_calls_is_pinned():
    text = _flat(_render("ci-cost"))
    assert "change set, candidate 120 0.59 192" in text
    assert "one candidate 246 1.13 427" in text


@cache
def _render(part):
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["gates", "--part", part]) == 0
    text = buf.getvalue()
    assert all(len(line) <= 78 for line in text.splitlines())
    return text


# ------------------------------------------------------------- the two policies


def _run(baseline, candidate, policy):
    return regression.evaluate(
        regression.load_policy(ROOT / "policies" / policy),
        ROOT / "runs" / baseline,
        ROOT / "runs" / candidate,
        HELD,
    )


def test_version_2_adds_one_check_and_keeps_version_1s_margin():
    v1 = regression.load_policy(ROOT / "policies/change_gate.yaml")
    v2 = regression.load_policy(ROOT / "policies/change_gate_v2.yaml")
    assert [c.id for c in v2.checks] == [
        "customer-check",
        "forbidden-actions",
        "tripwire",
        "not-worse-than-baseline",
        "cost-per-success",
        "latency-p95",
    ]
    kept = {c.id: c.margin_points for c in v1.checks if c.kind == "regression"}
    assert kept == {"not-worse-than-baseline": 10.0}
    assert {c.id: c.margin_points for c in v2.checks if c.id == "tripwire"} == {
        "tripwire": 5.0
    }


def test_reverting_the_change_is_held_by_version_1_and_blocked_by_version_2():
    one = _run("triage-gate-topics-heldout", "triage-metered-3-6", "change_gate.yaml")
    two = _run(
        "triage-gate-topics-heldout", "triage-metered-3-6", "change_gate_v2.yaml"
    )
    assert (one.status, one.exit_code) == ("HOLD", 2)
    assert (two.status, two.exit_code) == ("BLOCK", 1)


def test_the_change_itself_passes_on_the_held_out_tickets_with_a_latency_warning():
    report = _run(
        "triage-metered-3-6", "triage-gate-topics-heldout", "change_gate_v2.yaml"
    )
    assert report.status == "WARN" and report.exit_code == 0
    warned = [f.id for f in report.findings if f.status == "warn"]
    assert warned == ["latency-p95"]


def test_the_gate_refuses_to_compare_runs_from_different_models():
    report = _run("triage-metered-2-5", "triage-metered-3-6", "change_gate_v2.yaml")
    assert report.status == "BLOCK"
    assert "the models differ" in report.findings[0].lines[0]


def test_the_change_on_fresh_tickets_gains_forty_one_points_with_clean_hard_checks():
    report = regression.evaluate(
        regression.load_policy(ROOT / "policies/change_gate_v2.yaml"),
        ROOT / "runs/triage-change-base",
        ROOT / "runs/triage-change-topics",
        ROOT / "datasets/triage_change_v1.jsonl",
    )
    assert report.status == "PASS"
    by_id = {f.id: f for f in report.findings}
    assert "+40.8 points" in by_id["tripwire"].lines[0]
    assert by_id["customer-check"].status == "pass"
    assert by_id["forbidden-actions"].status == "pass"
