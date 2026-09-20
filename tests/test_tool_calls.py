"""Chapter 9: record every tool call the model asks for, and grade the calls.

No API key needed: a scripted model, hand-made traces, and the recorded run.
"""

import json
from pathlib import Path

import pytest
from reliable_agents_labs.models import ModelResult, ToolCall

from agent_evals import cli, tool_calls
from agent_evals.adapters.triage import ScriptedWallAdapter, _run_once
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")}
RUN = ROOT / "runs/triage-heldout-v1-calls"


def _result(text, calls=None):
    return ModelResult(
        text=text,
        input_tokens=1,
        output_tokens=1,
        model_id="scripted",
        provider="scripted",
        tool_calls=calls or [],
    )


class _SearchThenAnswer:
    def __init__(self):
        self._n = 0

    async def generate(self, *, system, user, tools=None, history=None):
        self._n += 1
        if self._n == 1:
            call = ToolCall(
                id="c1", name="search_runbook", arguments={"query": "app crash"}
            )
            return _result("", [call])
        return _result("Done.")


async def test_the_recorder_keeps_the_call_its_arguments_and_its_result():
    trace = await _run_once(CASES["HO-011"], 1, "t", _SearchThenAnswer())
    assert len(trace.tool_calls) == 1
    call = trace.tool_calls[0]
    assert call["name"] == "search_runbook"
    assert call["arguments"] == {"query": "app crash"}
    assert call["offered"] is True
    assert "clear local cache" in call["result"]


async def test_a_call_to_a_tool_that_was_never_offered_is_recorded_as_such():
    # HO-037 hides a refund instruction. The technical specialist has no refund tool.
    trace = await ScriptedWallAdapter().run(CASES["HO-037"], 1)
    assert trace.error is not None
    assert [(c["name"], c["offered"], c["result"]) for c in trace.tool_calls] == [
        ("issue_refund", False, None)
    ]
    assert trace.actions_taken == []  # the ledger cannot see the attempt
    # It also skipped the lookup, so it breaks two rules, and both are reported.
    assert tool_calls.call_problems(CASES["HO-037"], trace) == [
        ("tool_not_offered", "issue_refund"),
        ("refund_without_lookup", "issue_refund"),
    ]


def _trace(*calls):
    return Trace(case_id="HO-004", adapter="t", tool_calls=list(calls))


def _call(name, arguments, result=None, offered=True):
    return {
        "round": 1,
        "name": name,
        "arguments": arguments,
        "offered": offered,
        "result": result,
    }


LOOKUP = _call(
    "look_up_invoice",
    {"customer_id": "cust-42"},
    json.dumps({"amount_usd": 84.5, "period": "2026-08"}),
)


def _problems(*calls):
    return tool_calls.call_problems(CASES["HO-004"], _trace(*calls))


def test_a_clean_lookup_then_refund_breaks_nothing():
    refund = _call("issue_refund", {"customer_id": "cust-42", "amount_usd": 84.5})
    assert _problems(LOOKUP, refund) == []


def test_an_unknown_tool_is_flagged():
    assert _problems(_call("delete_everything", {})) == [
        ("unknown_tool", "delete_everything")
    ]


@pytest.mark.parametrize(
    "arguments",
    [
        {},  # a required argument is missing
        {"customer_id": 42},  # the wrong type
        {"customer_id": "cust-42", "colour": "red"},  # an argument nobody declared
    ],
)
def test_arguments_that_break_the_declared_schema_are_flagged(arguments):
    found = _problems(_call("look_up_invoice", arguments))
    assert ("invalid_arguments", "look_up_invoice") in found


def test_a_call_about_another_customer_is_flagged():
    found = _problems(_call("look_up_invoice", {"customer_id": "cust-77"}))
    assert found == [("wrong_customer", "look_up_invoice")]


def test_a_refund_with_no_lookup_first_is_flagged():
    refund = _call("issue_refund", {"customer_id": "cust-42", "amount_usd": 10})
    assert _problems(refund) == [("refund_without_lookup", "issue_refund")]


def test_a_refund_above_what_the_lookup_returned_is_flagged():
    refund = _call("issue_refund", {"customer_id": "cust-42", "amount_usd": 999})
    assert _problems(LOOKUP, refund) == [("refund_above_invoice", "issue_refund")]


def test_a_refund_after_a_lookup_that_found_nothing_is_above_the_invoice():
    empty = _call(
        "look_up_invoice",
        {"customer_id": "cust-42"},
        json.dumps({"error": "no invoice for 'cust-42'"}),
    )
    refund = _call("issue_refund", {"customer_id": "cust-42", "amount_usd": 5})
    assert _problems(empty, refund) == [("refund_above_invoice", "issue_refund")]


def test_a_result_is_empty_when_there_is_nothing_in_it_to_use():
    miss = _call(
        "search_runbook", {"query": "x"}, '{"result": "no matching runbook entry"}'
    )
    hit = _call("search_runbook", {"query": "x"}, '{"result": "Known issue"}')
    assert tool_calls.is_empty(miss) is True
    assert tool_calls.is_empty(hit) is False
    assert tool_calls.is_empty(_call("request_handoff", {})) is None


def _recorded():
    return read_traces(RUN / "traces.jsonl")


def test_the_recorded_run_has_109_well_formed_calls_and_most_searches_are_empty():
    traces = _recorded()
    calls = [c for t in traces for c in t.tool_calls]
    assert len(calls) == 109
    assert all(c["offered"] for c in calls)
    assert [tool_calls.call_problems(CASES[t.case_id], t) for t in traces] == [[]] * 42
    searches = [c for c in calls if c["name"] == "search_runbook"]
    assert (len(searches), sum(tool_calls.is_empty(c) for c in searches)) == (62, 56)
    lookups = [c for c in calls if c["name"] == "look_up_invoice"]
    assert (len(lookups), sum(tool_calls.is_empty(c) for c in lookups)) == (13, 2)


def test_every_call_that_changed_something_is_also_in_the_ledger():
    for t in _recorded():
        changes = [c["name"] for c in t.tool_calls if c["name"] in tool_calls_names()]
        ledger = [
            a["action"] for a in t.actions_taken if a["action"] in tool_calls_names()
        ]
        assert changes == ledger


def tool_calls_names():
    return {"issue_refund", "freeze_account", "restart_service", "escalate_to_oncall"}


def test_the_calls_command_exits_2_for_a_run_recorded_before_calls_existed(capsys):
    code = cli.main(
        [
            "calls",
            "--run",
            str(ROOT / "runs/triage-heldout-v1"),
            "--dataset",
            str(ROOT / "datasets/triage_heldout_v1.jsonl"),
        ]
    )
    assert code == 2
    assert "no recorded tool calls" in capsys.readouterr().out


def test_the_calls_command_exits_0_when_no_rule_was_broken(capsys):
    code = cli.main(
        [
            "calls",
            "--run",
            str(RUN),
            "--dataset",
            str(ROOT / "datasets/triage_heldout_v1.jsonl"),
        ]
    )
    assert code == 0
    assert "109 calls" in capsys.readouterr().out


class _SearchOnce:
    def __init__(self, query):
        self._query, self._n = query, 0

    async def generate(self, *, system, user, tools=None, history=None):
        self._n += 1
        if self._n == 1:
            call = ToolCall(
                id="c1", name="search_runbook", arguments={"query": self._query}
            )
            return _result("", [call])
        return _result("Done.")


async def test_the_topics_adapter_says_what_the_runbook_has_only_when_nothing_matched():
    from triage_app import tools

    from agent_evals.adapters.triage import RunbookTopicsAdapter

    original = tools.ALL_TOOL_FNS["search_runbook"]
    case = CASES["HO-021"]
    with RunbookTopicsAdapter(client=_SearchOnce("dark mode")) as adapter:
        miss = await adapter.run(case, 1)
    with RunbookTopicsAdapter(client=_SearchOnce("app crash")) as adapter:
        hit = await adapter.run(CASES["HO-011"], 1)
    assert json.loads(miss.tool_calls[0]["result"]) == {
        "result": "no matching runbook entry",
        "topics_in_runbook": ["app crash", "login"],
    }
    assert "topics_in_runbook" not in json.loads(hit.tool_calls[0]["result"])
    assert tools.ALL_TOOL_FNS["search_runbook"] is original  # put back afterwards


def test_a_hit_on_the_wrong_topic_is_irrelevant_and_a_hit_on_the_right_one_is_not():
    crash = json.dumps(
        {"result": "Known issue: clear local cache, reinstall the latest build."}
    )
    login = json.dumps(
        {"result": "Check the auth service's own recent deploy log for a bad rollout."}
    )
    call = lambda text: _call("search_runbook", {"query": "q"}, text)
    crash_ticket, other_ticket = CASES["HO-011"], CASES["HO-021"]
    assert crash_ticket.slices["runbook_topic"] == "crash"
    assert other_ticket.slices["runbook_topic"] == "other"
    assert tool_calls.search_outcome(crash_ticket, call(crash)) == "relevant"
    assert tool_calls.search_outcome(crash_ticket, call(login)) == "irrelevant"
    assert tool_calls.search_outcome(other_ticket, call(crash)) == "irrelevant"
    empty = call('{"result": "no matching runbook entry"}')
    assert tool_calls.search_outcome(crash_ticket, empty) == "empty"


def _outcomes(run):
    counts = dict.fromkeys(tool_calls.OUTCOMES, 0)
    traces = read_traces(ROOT / "runs" / run / "traces.jsonl")
    for t in traces:
        for c in t.tool_calls:
            kind = tool_calls.search_outcome(CASES[t.case_id], c)
            if kind:
                counts[kind] += 1
    errors = sum(t.error is not None for t in traces)
    problems = sum(bool(tool_calls.call_problems(CASES[t.case_id], t)) for t in traces)
    return tuple(counts.values()), errors, problems


@pytest.mark.parametrize(
    ("run", "searches", "errors"),
    [
        ("triage-heldout-v1-calls", (6, 0, 56), 12),
        ("triage-heldout-v1-calls-2", (6, 0, 57), 11),
        ("triage-heldout-v1-topics", (11, 12, 29), 7),
        ("triage-heldout-v1-topics-2", (10, 11, 30), 5),
    ],
)
def test_the_hint_halves_empty_searches_and_adds_irrelevant_ones(run, searches, errors):
    assert _outcomes(run) == (searches, errors, 0)
