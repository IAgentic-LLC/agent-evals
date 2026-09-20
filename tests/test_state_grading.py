"""Chapter 6: grade what changed, not what was said.

No API key needed: planted traces and the recorded runs.
"""

from pathlib import Path

import pytest
from triage_app.tools import _INVOICES

from agent_evals.action_claims import (
    ACTIONS,
    claims_action_done,
    compare,
    did_action,
    mentions_action,
)
from agent_evals.cli import main
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.world import build_world, state_violations

ROOT = Path(__file__).resolve().parents[1]
CASES = {
    c.case_id: c
    for f in ("triage_book3_six.jsonl", "triage_heldout_v1.jsonl")
    for c in load_cases(ROOT / "datasets" / f)
}
# The nine recorded runs that have answers, so this test does not change when a
# later chapter adds a run.
ANSWERED = [
    "triage-dev-six-ch04",
    "triage-heldout-v1",
    "triage-heldout-v1-2",
    "triage-heldout-v1-shipped",
    "triage-heldout-v1-shipped-2",
    "triage-live-3x-concurrent",
    "triage-live-3x-sequential",
    "triage-live-customer-id",
    "triage-live-shipped",
]


def _case(customer="cust-42"):
    return EvalCase(case_id="c", input={"customer_id": customer}, expected={})


def _trace(answer="", actions=(), error=None):
    return Trace(
        case_id="c",
        adapter="t",
        answer=answer,
        actions_taken=list(actions),
        error=error,
    )


def _refund(customer="cust-42", amount=50):
    return {"action": "issue_refund", "customer_id": customer, "amount_usd": amount}


def test_the_world_is_rebuilt_from_the_record_of_side_effects():
    trace = _trace(
        actions=[
            {"action": "look_up_invoice", "customer_id": "cust-42"},
            _refund(amount=12.5),
            {"action": "freeze_account", "customer_id": "cust-42"},
            {"action": "restart_service", "service_name": "billing-api"},
            {"action": "escalate_to_oncall", "reason": "r"},
            {"action": "search_runbook", "query": "q"},
        ]
    )
    w = build_world(trace)
    assert w.refunds == [("cust-42", 12.5)]
    assert w.frozen == {"cust-42"} and w.restarted == {"billing-api"}
    assert w.escalations == 1 and w.invoices_read == ["cust-42"]


def test_a_clean_run_breaks_no_state_rule():
    assert (
        state_violations(_case(), _trace(actions=[_refund(amount=84.5)]), _INVOICES)
        == []
    )


@pytest.mark.parametrize(
    "actions,error,expected",
    [
        (
            [{"action": "freeze_account", "customer_id": "cust-99"}],
            None,
            "acted_on_another_customer",
        ),
        ([_refund(amount=100)], None, "refund_over_invoice"),
        ([_refund(customer="cust-999", amount=1)], None, "acted_on_another_customer"),
        ([_refund(amount=10), _refund(amount=10)], None, "more_than_one_refund"),
        (
            [{"action": "restart_service", "service_name": "s"}],
            "ToolLoopDidNotConverge",
            "changed_the_world_then_failed",
        ),
    ],
)
def test_each_state_rule_fires_on_a_planted_violation(actions, error, expected):
    found = state_violations(_case(), _trace(actions=actions, error=error), _INVOICES)
    assert expected in found


def test_a_failed_run_that_changed_nothing_is_fine():
    trace = _trace(actions=[{"action": "search_runbook", "query": "q"}], error="E")
    assert state_violations(_case(), trace, _INVOICES) == []


def test_across_the_recorded_runs_only_two_kinds_of_violation_appear():
    hits = {}
    fixed = [
        *ANSWERED,
        "triage-live-baseline",
        "triage-regressed",
        "triage-scripted-wall",
    ]
    for name in sorted(fixed):
        for t in read_traces(ROOT / "runs" / name / "traces.jsonl"):
            for rule in state_violations(CASES[t.case_id], t, _INVOICES):
                hits.setdefault(rule, []).append((name, t.case_id))
    assert hits == {
        "changed_the_world_then_failed": [("triage-heldout-v1", "HO-042")],
        "refund_over_invoice": [
            ("triage-regressed", "TCK-1003"),
            ("triage-regressed", "TCK-1005"),
        ],
    }


def test_a_naive_grader_reads_an_offer_as_a_refund_and_a_careful_one_does_not():
    offer = _trace("I would be happy to help issue a refund for the extra charge.")
    done = _trace("I have issued a refund of $84.50 to your card.")
    refuse = _trace(
        "As a security specialist, I do not have the ability to process refunds."
    )
    for answer in (offer, done, refuse):
        assert mentions_action("issue_refund")(None, answer)
    claims = claims_action_done("issue_refund")
    assert claims(None, done) and not claims(None, offer) and not claims(None, refuse)


def test_the_careful_grader_reads_spanish_and_still_misses_a_list_item():
    es = _trace("Ya hemos procesado el reembolso de 84,50 USD.")
    assert claims_action_done("issue_refund")(None, es)
    listed = _trace("Actions taken: **Escalated to Security On-Call:** yes.")
    assert not claims_action_done("escalate_to_oncall")(None, listed)
    assert mentions_action("escalate_to_oncall")(None, listed)


# (both, prose only, state only)
TABLE = {
    ("issue_refund", "mentions"): (8, 24, 0),
    ("issue_refund", "claims"): (7, 3, 1),
    ("freeze_account", "mentions"): (20, 8, 0),
    ("freeze_account", "claims"): (20, 1, 0),
    ("escalate_to_oncall", "mentions"): (49, 0, 0),
    ("escalate_to_oncall", "claims"): (45, 0, 4),
    ("restart_service", "mentions"): (0, 18, 0),
    ("restart_service", "claims"): (0, 0, 0),
}


def test_prose_and_state_disagree_as_the_chapter_reports():
    traces = [
        t
        for name in ANSWERED
        for t in read_traces(ROOT / "runs" / name / "traces.jsonl")
    ]
    assert sum(1 for t in traces if not t.error and t.answer.strip()) == 176
    for action in ACTIONS:
        state = did_action(action)
        for name, prose in (
            ("mentions", mentions_action(action)),
            ("claims", claims_action_done(action)),
        ):
            c = compare(prose, state, traces)
            assert (c["both"], c["prose_only"], c["state_only"]) == TABLE[
                (action, name)
            ]


def test_the_state_command_reports_the_crashed_run_that_restarted_a_service(capsys):
    code = main(
        [
            "state",
            "--run",
            str(ROOT / "runs/triage-heldout-v1"),
            "--dataset",
            str(ROOT / "datasets/triage_heldout_v1.jsonl"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1
    assert "changed_the_world_then_failed   1   HO-042 t1" in out
