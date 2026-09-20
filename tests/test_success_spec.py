"""Chapter 2: success written as required work, not only correct routing.

No API key needed: the recorded runs and a capturing scripted model.
"""

from pathlib import Path

from agent_evals.adapters.triage import CustomerIdAdapter
from agent_evals.graders import required_actions_missing
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.scorecard import build_scorecard

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "triage_book3_six.jsonl"


def _case(required):
    return EvalCase(
        case_id="c",
        input={},
        expected={"handled_by": "billing", "required_actions": required},
    )


def _trace(actions):
    return Trace(
        case_id="c",
        adapter="t",
        handled_by="billing",
        actions_taken=[{"action": a} for a in actions],
    )


def test_a_missing_required_action_is_named():
    assert required_actions_missing(_case(["look_up_invoice"]), _trace([])) == [
        "look_up_invoice"
    ]


def test_a_taken_required_action_is_not_missing_even_with_extras():
    trace = _trace(["look_up_invoice", "freeze_account"])
    assert required_actions_missing(_case(["look_up_invoice"]), trace) == []


def test_a_case_that_lists_no_required_actions_has_nothing_missing():
    assert required_actions_missing(_case([]), _trace([])) == []


def _card(run):
    return build_scorecard(
        "six",
        run,
        load_cases(DATASET),
        read_traces(ROOT / "runs" / run / "traces.jsonl"),
    )


def test_the_chapter_one_recording_passes_routing_but_only_four_of_six_do_the_work():
    card = _card("triage-live-baseline")
    assert card.routing_successes == 6
    assert card.actions_successes == 4
    assert round(card.actions_interval.low, 3) == 0.300
    assert round(card.actions_interval.high, 3) == 0.903


def test_the_shipped_run_misses_the_invoice_lookup_on_both_billing_tickets():
    cases = {c.case_id: c for c in load_cases(DATASET)}
    traces = read_traces(ROOT / "runs" / "triage-live-shipped" / "traces.jsonl")
    missed = {
        t.case_id for t in traces if required_actions_missing(cases[t.case_id], t)
    }
    assert missed == {"TCK-1001", "TCK-1006"}


def test_passing_on_the_customer_id_lets_all_six_do_the_work():
    card = _card("triage-live-customer-id")
    assert (card.routing_successes, card.actions_successes) == (6, 6)


class _CapturingClient:
    def __init__(self):
        self.questions = []

    async def generate(self, *, system, user, tools=None, history=None):
        from reliable_agents_labs.models import ModelResult

        self.questions.append(user)
        return ModelResult(
            text="ok",
            input_tokens=1,
            output_tokens=1,
            model_id="scripted",
            provider="scripted",
            tool_calls=[],
        )


async def test_the_customer_id_adapter_puts_the_id_in_the_question_and_restores_it():
    from triage_app import specialists

    before = specialists._question_for
    client = _CapturingClient()
    case = next(c for c in load_cases(DATASET) if c.case_id == "TCK-1001")
    await CustomerIdAdapter(client=client).run(case, 1)
    assert client.questions[0].startswith("Customer ID: cust-42")
    assert specialists._question_for is before
