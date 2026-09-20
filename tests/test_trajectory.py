"""Chapter 10: constraints on the shape of a run, and a guard that enforces one.

No API key needed: hand-made traces, a scripted model, and the recorded runs.
"""

import json
from pathlib import Path

import pytest
from reliable_agents_labs.models import ModelResult, ToolCall

from agent_evals import cli, trajectory
from agent_evals.adapters.triage import StallGuardAdapter
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")}
MISS = '{"result": "no matching runbook entry"}'
HIT = json.dumps(
    {"result": "Known issue: clear local cache, reinstall the latest build."}
)


def _call(name, arguments, result=None, round_=1):
    return {
        "round": round_,
        "name": name,
        "arguments": arguments,
        "offered": True,
        "result": result,
    }


def _search(query, result, round_=1):
    return _call("search_runbook", {"query": query}, result, round_)


def _trace(*calls, error=None):
    return Trace(case_id="HO-011", adapter="t", tool_calls=list(calls), error=error)


def _found(trace, max_calls=3, max_stall=3):
    # HO-011 is a crash ticket, so the crash entry is a relevant result for it.
    return trajectory.violations(CASES["HO-011"], trace, max_calls, max_stall)


def test_a_short_run_that_ends_in_an_answer_breaks_nothing():
    assert _found(_trace(_search("dark mode", MISS))) == []


def test_a_run_that_ended_in_an_error_is_flagged():
    assert _found(_trace(_search("q", MISS), error="ToolLoopDidNotConverge")) == [
        "ended_without_answer"
    ]


def test_more_calls_than_the_budget_is_flagged():
    calls = [_search(f"q{i}", HIT, i) for i in range(4)]
    assert _found(_trace(*calls)) == ["over_call_budget"]
    assert _found(_trace(*calls), max_calls=4) == []


def test_three_empty_results_in_a_row_is_a_stall_and_a_hit_in_between_is_not():
    stall = [_search(f"q{i}", MISS, i) for i in range(3)]
    assert "stalled" in _found(_trace(*stall))
    broken = [_search("a", MISS, 1), _search("b", MISS, 2), _search("c", HIT, 3)]
    assert _found(_trace(*broken, _search("d", MISS, 4))) == ["over_call_budget"]


def test_the_same_call_twice_is_flagged():
    twice = [_search("dark mode", MISS, 1), _search("dark mode", MISS, 2)]
    assert _found(_trace(*twice)) == ["repeated_call"]


def test_a_ticket_handed_back_to_a_specialist_that_had_it_is_flagged():
    case = CASES["HO-039"]  # a technical ticket
    go = _call("request_handoff", {"target_category": "billing", "reason": "r"})
    back = _call("request_handoff", {"target_category": "technical", "reason": "r"})
    trace = Trace(case_id="HO-039", adapter="t", tool_calls=[go, back])
    assert trajectory.handoff_path(case, trace) == ["technical", "billing", "technical"]
    assert trajectory.violations(case, trace, 3, 3) == ["handoff_returned"]
    one_way = Trace(case_id="HO-039", adapter="t", tool_calls=[go])
    assert trajectory.violations(case, one_way, 3, 3) == []


def test_empty_rounds_count_rounds_and_a_hit_resets_the_count():
    misses = [_search("a", MISS, 1), _search("b", MISS, 2), _search("c", MISS, 3)]
    assert trajectory.empty_rounds_in_a_row(_trace(*misses)) == 3
    reset = [_search("a", MISS, 1), _search("b", HIT, 2), _search("c", MISS, 3)]
    assert trajectory.empty_rounds_in_a_row(_trace(*reset)) == 1
    # Two empty results in one round are one empty round, not two.
    together = [_search("a", MISS, 1), _search("b", MISS, 1)]
    assert trajectory.empty_rounds_in_a_row(_trace(*together)) == 1


def _model_result(text, calls=None):
    return ModelResult(
        text=text,
        input_tokens=1,
        output_tokens=1,
        model_id="scripted",
        provider="scripted",
        tool_calls=calls or [],
    )


class _Script:
    """Replies from a list, and remembers what it was asked."""

    def __init__(self, replies):
        self._replies = iter(replies)
        self.asked = []

    async def generate(self, *, system, user, tools=None, history=None):
        self.asked.append({"system": system, "tools": tools})
        return next(self._replies)


def _search_call(query, n):
    return _model_result(
        "", [ToolCall(id=f"c{n}", name="search_runbook", arguments={"query": query})]
    )


async def test_the_guard_stops_offering_tools_after_three_empty_rounds():
    script = _Script(
        [
            _search_call("dark mode", 1),
            _search_call("theme", 2),
            _search_call("contrast", 3),
            _model_result("Here is what I know. I could not check the runbook."),
        ]
    )
    with StallGuardAdapter(client=script) as adapter:
        trace = await adapter.run(CASES["HO-021"], 1)
    assert trace.error is None and trace.answer.startswith("Here is what I know")
    assert len(trace.tool_calls) == 3
    assert [a["tools"] is None for a in script.asked] == [False, False, False, True]
    assert "Do not call another tool" in script.asked[-1]["system"]
    assert "Do not call another tool" not in script.asked[0]["system"]


async def test_the_guard_does_nothing_when_a_useful_result_breaks_the_streak():
    script = _Script(
        [
            _search_call("dark mode", 1),
            _search_call("theme", 2),
            _search_call("app crash", 3),
            _search_call("contrast", 4),
            _model_result("Done."),
        ]
    )
    with StallGuardAdapter(client=script) as adapter:
        trace = await adapter.run(CASES["HO-021"], 1)
    assert trace.error is None and len(trace.tool_calls) == 4
    assert all(a["tools"] is not None for a in script.asked)


async def test_the_guard_still_lets_a_run_fail_when_hits_keep_resetting_it():
    # A hit after every two misses resets the count, so the guard never fires and the
    # product's own round limit ends the run: the gap the chapter reports.
    replies = [
        _search_call("a", 1),
        _search_call("b", 2),
        _search_call("app crash", 3),
        _search_call("c", 4),
        _search_call("d", 5),
        _search_call("e", 6),
    ]
    with StallGuardAdapter(client=_Script(replies)) as adapter:
        trace = await adapter.run(CASES["HO-021"], 1)
    assert trace.error is not None and trace.error.startswith("ToolLoopDidNotConverge")


def test_the_guard_puts_the_products_loop_back_afterwards():
    from triage_app import specialists

    before = specialists.run_tool_loop
    with StallGuardAdapter(client=_Script([])):
        assert specialists.run_tool_loop is not before
    assert specialists.run_tool_loop is before


RUNS = {
    "unchanged": ("triage-heldout-v1-calls", "triage-heldout-v1-calls-2"),
    "hint": ("triage-heldout-v1-topics", "triage-heldout-v1-topics-2"),
    "guard": ("triage-heldout-v1-guard", "triage-heldout-v1-guard-2"),
}


def _counts(run):
    traces = read_traces(ROOT / "runs" / run / "traces.jsonl")
    counts = dict.fromkeys(trajectory.RULES, 0)
    for t in traces:
        for rule in trajectory.violations(CASES[t.case_id], t, 3, 3):
            counts[rule] += 1
    return tuple(counts.values())


@pytest.mark.parametrize(
    ("run", "expected"),
    [
        ("triage-heldout-v1-calls", (12, 13, 11, 0, 3)),
        ("triage-heldout-v1-calls-2", (11, 14, 12, 0, 2)),
        ("triage-heldout-v1-topics", (7, 9, 7, 0, 2)),
        ("triage-heldout-v1-topics-2", (5, 8, 8, 0, 2)),
        ("triage-heldout-v1-guard", (2, 3, 12, 0, 1)),
        ("triage-heldout-v1-guard-2", (3, 3, 11, 0, 1)),
    ],
)
def test_the_recorded_runs_break_the_constraints_this_often(run, expected):
    assert _counts(run) == expected


def test_a_stall_limit_of_three_stops_17_failed_runs_and_interrupts_6_answers():
    traces = [
        t
        for run in RUNS["unchanged"]
        for t in read_traces(ROOT / "runs" / run / "traces.jsonl")
    ]
    failed = [t for t in traces if t.error]
    answered = [t for t in traces if not t.error]
    stopped = sum(trajectory.empty_rounds_in_a_row(t) >= 3 for t in failed)
    cut = sum(trajectory.empty_rounds_in_a_row(t) >= 3 for t in answered)
    assert (len(failed), len(answered), stopped, cut) == (23, 61, 17, 6)


def test_the_guard_runs_cut_errors_from_23_of_84_to_5_of_84():
    def errors(runs):
        return sum(
            t.error is not None
            for run in runs
            for t in read_traces(ROOT / "runs" / run / "traces.jsonl")
        )

    assert (errors(RUNS["unchanged"]), errors(RUNS["guard"])) == (23, 5)


def test_the_trajectory_command_needs_its_limits_and_exits_1_when_one_broke(capsys):
    args = [
        "trajectory",
        "--run",
        str(ROOT / "runs/triage-heldout-v1-calls"),
        "--dataset",
        str(ROOT / "datasets/triage_heldout_v1.jsonl"),
    ]
    with pytest.raises(SystemExit):
        cli.main(args)  # the limits are required
    assert cli.main(args + ["--max-calls", "3", "--max-stall", "3"]) == 1
    assert "16 of 42" in capsys.readouterr().out
