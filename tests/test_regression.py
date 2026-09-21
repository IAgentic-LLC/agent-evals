"""Chapter 23: a candidate run against a baseline run, with a risk tier.

The decision function is checked on hand-made differences, the requirements on small
synthetic runs, and the whole gate on the recorded runs of chapter 22, where the size of
the difference is known.
"""

import json
from pathlib import Path

import pytest

from agent_evals import cli, regression
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[1]
HELD = ROOT / "datasets/triage_heldout_v1.jsonl"
POLICY = ROOT / "policies/change_gate.yaml"


def _case(case_id, customer="cust-1"):
    return EvalCase(
        case_id=case_id,
        input={"customer_id": customer, "category": "technical"},
        expected={"handled_by": "technical", "required_actions": ["search_runbook"]},
    )


def _met(case_id, trial=1):
    return Trace(
        case_id=case_id,
        trial=trial,
        adapter="a",
        actions_taken=[{"action": "search_runbook"}],
    )


def _miss(case_id, trial=1):
    return Trace(case_id=case_id, trial=trial, adapter="a", actions_taken=[])


def _write_run(path, traces, sha="abc", model="m1", trials=3):
    path.mkdir(parents=True)
    (path / "traces.jsonl").write_text(
        "".join(t.model_dump_json() + "\n" for t in traces), encoding="utf8"
    )
    manifest = {
        "dataset": {"sha256": sha, "cases": len({t.case_id for t in traces})},
        "trials": trials,
    }
    if model:
        manifest["model"] = {"provider": "gemini", "model_id": model}
    (path / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    return path


def _dataset(path, cases):
    path.write_text("".join(c.model_dump_json() + "\n" for c in cases), encoding="utf8")
    return path


def _policy(**overrides):
    policy = regression.load_policy(POLICY)
    for key, value in overrides.items():
        setattr(policy.require, key, value)
    return policy


# ------------------------------------------------------------------ decide


def test_a_run_that_differs_by_nothing_passes_every_rule():
    diffs = [0.0] * 40
    for method in ("point", "sign", "interval"):
        assert regression.decide(diffs, method, 5)[0] == "pass"


def test_a_large_drop_on_every_ticket_is_blocked_by_every_rule():
    diffs = [-0.5] * 40
    for method in ("point", "sign", "interval"):
        assert regression.decide(diffs, method, 5)[0] == "block"


def test_a_noisy_small_drop_is_unsettled_by_the_interval_rule_and_blocked_by_the_point_rule():
    diffs = [-1 / 3] * 6 + [1 / 3] * 5 + [0.0] * 29
    assert regression.decide(diffs, "point", 1)[0] == "pass"
    assert regression.decide([-0.2] * 8 + [0.3] * 3 + [0.0] * 29, "point", 1)[0] == (
        "block"
    )
    assert regression.decide(diffs, "interval", 1)[0] == "inconclusive"


def test_the_interval_rule_gets_stricter_as_the_margin_gets_smaller():
    diffs = [-0.3] * 5 + [0.0] * 35
    verdicts = [regression.decide(diffs, "interval", m)[0] for m in (15, 10, 5, 1)]
    order = {"pass": 0, "inconclusive": 1, "block": 2}
    assert [order[v] for v in verdicts] == sorted(order[v] for v in verdicts)
    assert verdicts[0] == "pass" and verdicts[-1] != "pass"


def test_the_sign_rule_blocks_only_when_worse_and_significant():
    assert regression.decide([-0.1] * 10 + [0.0] * 30, "sign", 0)[0] == "block"
    assert regression.decide([-0.1] * 3 + [0.0] * 37, "sign", 0)[0] == "pass"
    assert regression.decide([0.1] * 10 + [0.0] * 30, "sign", 0)[0] == "pass"


def test_the_interval_can_be_passed_in_so_that_several_margins_share_one_draw():
    diffs = [-0.05] * 20 + [0.0] * 20
    low, high = -0.09, 0.0
    assert regression.decide(diffs, "interval", 10, interval=(low, high))[0] == "pass"
    assert regression.decide(diffs, "interval", 5, interval=(low, high))[0] == (
        "inconclusive"
    )


# ------------------------------------------------------------ ticket differences


def test_ticket_differences_are_candidate_minus_baseline_shares_per_ticket():
    cases = {c: _case(c) for c in ("T1", "T2")}
    base = [_met("T1", 1), _met("T1", 2), _miss("T2", 1), _miss("T2", 2)]
    cand = [_met("T1", 1), _miss("T1", 2), _met("T2", 1), _met("T2", 2)]
    assert regression.ticket_differences(cases, base, cand) == [-0.5, 1.0]


# ------------------------------------------------------------------ evaluate


def _fixture(tmp_path, base_traces, cand_traces, **manifest):
    cases = [_case(f"T{i}") for i in range(40)]
    dataset = _dataset(tmp_path / "d.jsonl", cases)
    base = _write_run(tmp_path / "base", base_traces)
    cand = _write_run(tmp_path / "cand", cand_traces, **manifest)
    return dataset, base, cand


def _all(kind, trial=1):
    return [(_met if kind == "met" else _miss)(f"T{i}", trial) for i in range(40)]


def test_two_runs_with_different_dataset_files_are_not_compared(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("met"), sha="other")
    report = regression.evaluate(_policy(), base, cand, dataset)
    assert report.status == "BLOCK"
    assert "different dataset files" in report.findings[0].lines[0]


def test_two_runs_with_different_models_are_not_compared(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("met"), model="m2")
    report = regression.evaluate(_policy(), base, cand, dataset)
    assert report.status == "BLOCK"
    assert "the models differ: m1 and m2" in report.findings[0].lines[0]


def test_a_run_that_did_not_record_its_model_is_not_compared(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("met"), model=None)
    assert (
        "did not record"
        in regression.evaluate(_policy(), base, cand, dataset).findings[0].lines[0]
    )


def test_too_few_tickets_is_a_problem_before_any_statistics_run(tmp_path):
    cases = [_case(f"T{i}") for i in range(5)]
    dataset = _dataset(tmp_path / "d.jsonl", cases)
    traces = [_met(f"T{i}") for i in range(5)]
    base = _write_run(tmp_path / "b", traces)
    cand = _write_run(tmp_path / "c", traces)
    report = regression.evaluate(_policy(), base, cand, dataset)
    assert "only 5 tickets" in report.findings[0].lines[0]


def test_an_unchanged_run_passes_and_a_run_that_lost_every_ticket_is_blocked(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("met"))
    assert regression.evaluate(_policy(), base, cand, dataset).status == "PASS"
    (tmp_path / "x").mkdir()
    dataset, base, cand = _fixture(tmp_path / "x", _all("met"), _all("miss"))
    report = regression.evaluate(_policy(), base, cand, dataset)
    assert report.status == "BLOCK" and report.exit_code == 1


def test_a_run_that_acts_on_another_customer_is_blocked_whatever_its_met_rate(tmp_path):
    bad = _all("met")
    bad[0] = Trace(
        case_id="T0",
        trial=1,
        adapter="a",
        actions_taken=[
            {"action": "search_runbook"},
            {"action": "look_up_invoice", "customer_id": "cust-99"},
        ],
    )
    dataset, base, cand = _fixture(tmp_path, _all("met"), bad)
    report = regression.evaluate(_policy(), base, cand, dataset)
    hard = {f.id: f for f in report.findings}["customer-check"]
    assert hard.status == "block" and report.status == "BLOCK"
    assert "T0 run 1" in hard.lines


def test_the_low_tier_runs_no_checks_and_the_medium_tier_only_the_hard_ones(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("miss"))
    low = regression.evaluate(_policy(), base, cand, dataset, tier="low")
    assert low.status == "PASS" and [f.id for f in low.findings] == [
        "no-checks-at-this-tier"
    ]
    medium = regression.evaluate(_policy(), base, cand, dataset, tier="medium")
    assert {f.kind for f in medium.findings} == {"hard"}
    assert medium.status == "PASS"


def test_an_unsettled_result_holds_and_the_policy_can_change_what_that_means(tmp_path):
    base = [_met(f"T{i}", t) for i in range(40) for t in (1, 2, 3)]
    cand = [
        (_miss if i < 6 else _met)(f"T{i}", t) for i in range(40) for t in (1, 2, 3)
    ]
    dataset, b, c = _fixture(tmp_path, base, cand)
    policy = _policy()
    for check in policy.checks:
        if check.kind == "regression":
            check.margin_points = 5
    report = regression.evaluate(policy, b, c, dataset)
    assert report.status == "HOLD" and report.exit_code == 2
    for check in policy.checks:
        if check.kind == "regression":
            check.on_inconclusive = "pass"
    assert regression.evaluate(policy, b, c, dataset).status == "PASS"


def test_a_budget_over_its_limit_warns_and_does_not_block(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("met"))
    slow = [
        t.model_copy(update={"latency_s": 30.0})
        for t in regression.read_traces(cand / "traces.jsonl")
    ]
    fast = [
        t.model_copy(update={"latency_s": 10.0})
        for t in regression.read_traces(base / "traces.jsonl")
    ]
    _write_run(tmp_path / "s", slow)
    _write_run(tmp_path / "f", fast)
    report = regression.evaluate(_policy(), tmp_path / "f", tmp_path / "s", dataset)
    assert report.status == "WARN" and report.exit_code == 0


def test_the_markdown_summary_names_the_result_and_every_check(tmp_path):
    dataset, base, cand = _fixture(tmp_path, _all("met"), _all("miss"))
    text = regression.render_markdown(
        regression.evaluate(_policy(), base, cand, dataset)
    )
    assert "BLOCK" in text and "not-worse-than-baseline" in text


# ------------------------------------------------------------------ risk tiers


def test_book_3s_release_gate_puts_a_change_to_the_tools_in_the_high_tier():
    assert regression.tier_for_files(["backend/src/triage_app/tools.py"]) == "high"
    assert regression.tier_for_files(["frontend/src/App.tsx"]) == "low"
    assert (
        regression.tier_for_files(
            ["frontend/src/App.tsx", "backend/src/triage_app/tools.py"]
        )
        == "high"
    )


# ------------------------------------------------------------ the recorded runs


def _runs(baseline, candidate):
    return regression.evaluate(
        _policy(),
        ROOT / "runs" / baseline,
        ROOT / "runs" / candidate,
        HELD,
    )


def test_a_gain_of_21_points_passes_and_the_same_pair_reversed_is_blocked():
    forward = _runs("triage-metered-3-6", "triage-metered-3-6-guard")
    assert forward.status == "PASS"
    back = _runs("triage-metered-3-6-guard", "triage-metered-3-6")
    assert back.status == "BLOCK" and back.exit_code == 1
    check = {f.id: f for f in back.findings}["not-worse-than-baseline"]
    assert "-21.4 points" in check.lines[0]


def test_a_condition_compared_with_its_own_earlier_run_is_not_blocked():
    report = _runs("triage-metered-3-6-guard", "triage-cost-3-6-guard")
    assert report.status in ("PASS", "WARN")


def test_the_report_fits_the_page_and_the_cli_returns_the_exit_code(capsys):
    argv = [
        "regress",
        "--baseline",
        str(ROOT / "runs/triage-metered-3-6-guard"),
        "--candidate",
        str(ROOT / "runs/triage-metered-3-6"),
        "--dataset",
        str(HELD),
        "--policy",
        str(POLICY),
    ]
    assert cli.main(argv) == 1
    out = capsys.readouterr().out
    assert all(len(line) <= 78 for line in out.splitlines())
    assert out.rstrip().endswith("RESULT: BLOCK")
    argv[argv.index("--policy") + 1] = str(POLICY)
    assert cli.main([*argv, "--changed-file", "frontend/src/App.tsx"]) == 0


def test_the_policy_file_loads_and_says_who_owns_the_margin():
    policy = regression.load_policy(POLICY)
    kinds = [c.kind for c in policy.checks]
    assert kinds == ["hard", "hard", "regression", "budget", "budget"]
    assert "product owner's decision" in POLICY.read_text(encoding="utf8")
    with pytest.raises(Exception):  # noqa: B017
        regression.load_policy(ROOT / "README.md")
