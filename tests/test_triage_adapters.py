"""No API key needed: the scripted adapters and the recorded live run."""

from pathlib import Path

from agent_evals.adapters.triage import RegressedAdapter, ScriptedWallAdapter
from agent_evals.cli import main
from agent_evals.gate import evaluate, load_policy
from agent_evals.runner import load_cases, read_traces, run_cases
from agent_evals.scorecard import build_scorecard

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "triage_book3_six.jsonl"
BASELINE = ROOT / "runs" / "triage-live-baseline"


async def _card(adapter):
    cases = load_cases(DATASET)
    traces = await run_cases(cases, adapter)
    return build_scorecard("six", adapter.name, cases, traces), traces


async def test_the_wall_holds_and_fails_closed_on_the_injected_tickets():
    card, traces = await _card(ScriptedWallAdapter())
    assert card.invariant_violations == 0
    injected = [t for t in traces if t.case_id in {"TCK-1003", "TCK-1005"}]
    assert len(injected) == 2 and all(t.error for t in injected)


async def test_the_regression_keeps_routing_perfect_but_breaks_the_invariant():
    card, _ = await _card(RegressedAdapter())
    assert card.routing_successes == 6  # routing quality cannot see it
    assert card.invariant_violations == 2
    assert card.violated_cases == ["TCK-1003", "TCK-1005"]


async def test_the_original_quality_rule_passes_the_regression_and_the_hard_gate_blocks_it():
    card, _ = await _card(RegressedAdapter())
    result = evaluate(load_policy(ROOT / "policies" / "book3_original.yaml"), card)
    by_id = {r.id: r for r in result.results}
    assert by_id["routing-point-estimate"].passed
    assert not by_id["no-forbidden-actions"].passed
    assert not result.passed


def test_the_recorded_live_baseline_reproduces_six_for_six_with_its_interval():
    cases = load_cases(DATASET)
    card = build_scorecard(
        "six", "baseline", cases, read_traces(BASELINE / "traces.jsonl")
    )
    assert (card.routing_successes, card.observations) == (6, 6)
    assert card.invariant_violations == 0
    assert round(card.routing_interval.low, 3) == 0.610


def test_the_cli_gate_exit_codes(tmp_path):
    base = ["--run", str(BASELINE), "--dataset", str(DATASET)]
    assert (
        main(["gate", *base, "--policy", str(ROOT / "policies/book3_original.yaml")])
        == 0
    )
    assert (
        main(["gate", *base, "--policy", str(ROOT / "policies/interval_aware.yaml")])
        == 1
    )
    regressed = tmp_path / "regressed"
    main(
        [
            "run",
            "--adapter",
            "triage-regressed",
            "--dataset",
            str(DATASET),
            "--out",
            str(regressed),
        ]
    )
    args = ["--run", str(regressed), "--dataset", str(DATASET)]
    assert (
        main(["gate", *args, "--policy", str(ROOT / "policies/book3_original.yaml")])
        == 1
    )
