from pathlib import Path

from agent_evals.gate import evaluate, load_policy
from agent_evals.schema import EvalCase, Trace
from agent_evals.scorecard import build_scorecard

POLICIES = Path(__file__).resolve().parents[1] / "policies"


def _run(n: int, misrouted: int = 0, violations: int = 0):
    """n cases; the first `misrouted` are routed wrong, the first `violations` break an invariant."""
    cases, traces = [], []
    for i in range(n):
        cases.append(
            EvalCase(
                case_id=f"c{i}",
                input={},
                expected={"handled_by": "billing"},
                invariants={"forbidden_actions": ["issue_refund"]},
            )
        )
        traces.append(
            Trace(
                case_id=f"c{i}",
                adapter="test",
                handled_by="technical" if i < misrouted else "billing",
                actions_taken=[{"action": "issue_refund"}] if i < violations else [],
            )
        )
    return build_scorecard("d", "r", cases, traces)


def _gate(policy: str, scorecard):
    return evaluate(load_policy(POLICIES / policy), scorecard)


def test_the_original_rule_lets_one_misroute_in_six_through():
    result = _gate("book3_original.yaml", _run(6, misrouted=1))
    assert result.passed  # 5/6 = 83.3% clears the 80% point-estimate rule


def test_the_original_rule_blocks_two_misroutes_in_six():
    result = _gate("book3_original.yaml", _run(6, misrouted=2))
    assert not result.passed


def test_an_interval_aware_gate_cannot_be_passed_with_six_cases():
    result = _gate("interval_aware.yaml", _run(6))
    assert not result.passed
    quality = next(r for r in result.results if r.kind == "quality")
    assert "at least 16" in quality.note


def test_an_interval_aware_gate_can_be_passed_with_enough_cases():
    assert _gate("interval_aware.yaml", _run(16)).passed


def test_one_invariant_violation_blocks_even_a_perfect_quality_score():
    for policy in ("book3_original.yaml", "interval_aware.yaml"):
        result = _gate(policy, _run(20, violations=1))
        assert not result.passed
        failed = [r.id for r in result.results if not r.passed]
        assert failed == [
            "no-forbidden-actions"
        ]  # quality passed; only the hard gate failed
