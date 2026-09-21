"""Chapter 25: the two fixes for the customer-id incident, checked without a key.

The guard is a tool wrapper, so it can be tested by calling the patched tools directly. The
note is a change to the question, so it is tested by building a question.
"""

import json

from triage_app import specialists, tools
from triage_app.tickets import Ticket

from agent_evals.adapters.triage import (
    ID_NOTE,
    CustomerGuardAdapter,
    CustomerIdNoteAdapter,
)


def _ticket(customer="cust-42"):
    return Ticket(
        ticket_id="T-1",
        customer_id=customer,
        category="billing",
        subject="Invoice",
        body="Please pull up my invoice.",
    )


def test_the_guard_refuses_another_customer_id_and_records_no_action():
    tools.ACTIONS_TAKEN.clear()
    with CustomerGuardAdapter():
        specialists._question_for(_ticket(), None)
        reply = json.loads(
            tools.ALL_TOOL_FNS["look_up_invoice"]({"customer_id": "cust_42"})
        )
    assert "own customer" in reply["error"] and "cust-42" in reply["error"]
    assert list(tools.ACTIONS_TAKEN) == []


def test_the_guard_lets_the_tickets_own_customer_through():
    tools.ACTIONS_TAKEN.clear()
    with CustomerGuardAdapter():
        specialists._question_for(_ticket(), None)
        reply = json.loads(
            tools.ALL_TOOL_FNS["look_up_invoice"]({"customer_id": "cust-42"})
        )
    assert reply["amount_usd"] == 84.5
    assert list(tools.ACTIONS_TAKEN) == [
        {"action": "look_up_invoice", "customer_id": "cust-42"}
    ]


def test_the_guard_covers_every_tool_that_takes_a_customer_id():
    with CustomerGuardAdapter():
        specialists._question_for(_ticket(), None)
        tools.ACTIONS_TAKEN.clear()
        for name, args in (
            ("issue_refund", {"customer_id": "cust-99", "amount_usd": 5}),
            ("freeze_account", {"customer_id": "cust-99"}),
        ):
            assert "error" in json.loads(tools.ALL_TOOL_FNS[name](args))
    assert list(tools.ACTIONS_TAKEN) == []


def test_the_guard_leaves_a_tool_without_a_customer_id_alone():
    with CustomerGuardAdapter():
        specialists._question_for(_ticket(), None)
        reply = json.loads(tools.ALL_TOOL_FNS["search_runbook"]({"query": "app crash"}))
    assert "result" in reply


def test_the_guard_puts_the_original_tools_and_question_back():
    before = {n: tools.ALL_TOOL_FNS[n] for n in tools.ALL_TOOL_FNS}
    question = specialists._question_for
    with CustomerGuardAdapter():
        assert tools.ALL_TOOL_FNS["look_up_invoice"] is not before["look_up_invoice"]
    assert {n: tools.ALL_TOOL_FNS[n] for n in tools.ALL_TOOL_FNS} == before
    assert specialists._question_for is question


def test_the_note_is_added_to_the_question_and_removed_afterwards():
    question = specialists._question_for
    with CustomerIdNoteAdapter():
        text = specialists._question_for(_ticket("cust-311"), None)
    assert text.startswith(ID_NOTE)
    assert "Customer ID: cust-311" in text
    assert specialists._question_for is question
    assert not specialists._question_for(_ticket(), None).startswith(ID_NOTE)
