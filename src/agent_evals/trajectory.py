"""Constraints on the shape of a whole run: how long, how repetitive, how it ended.

Chapter 9 graded each tool call. This module grades the sequence. It does not say what
the right path is, because many paths reach the same good result. It says what no
good path does: run past a budget, repeat itself, stall on empty results, hand a
ticket back to a specialist that already had it, or end without an answer.
"""

import json
from typing import Any

from agent_evals import tool_calls
from agent_evals.schema import EvalCase, Trace

RULES = (
    "ended_without_answer",
    "over_call_budget",
    "stalled",
    "repeated_call",
    "handoff_returned",
)


def handoff_path(case: EvalCase, trace: Trace) -> list[str]:
    """The specialists that had the ticket, in order, starting with its category."""
    path = [case.input["category"]]
    for call in trace.tool_calls:
        if call["name"] == "request_handoff":
            path.append(call["arguments"].get("target_category", "?"))
    return path


def unproductive(case: EvalCase, call: dict[str, Any]) -> bool:
    """A call that gave the agent nothing to work with."""
    if call["name"] == "search_runbook":
        return tool_calls.search_outcome(case, call) in ("empty", "irrelevant")
    return tool_calls.is_empty(call) is True


def longest_stall(case: EvalCase, trace: Trace) -> int:
    """The most calls in a row that gave nothing to work with."""
    best = run = 0
    for call in trace.tool_calls:
        if call["result"] is None:
            continue
        run = run + 1 if unproductive(case, call) else 0
        best = max(best, run)
    return best


def violations(
    case: EvalCase, trace: Trace, max_calls: int, max_stall: int
) -> list[str]:
    """The constraints this run broke, once each, in the order of RULES."""
    found = []
    if trace.error is not None:
        found.append("ended_without_answer")
    if len(trace.tool_calls) > max_calls:
        found.append("over_call_budget")
    if longest_stall(case, trace) >= max_stall:
        found.append("stalled")
    seen = set()
    for call in trace.tool_calls:
        key = (call["name"], json.dumps(call["arguments"], sort_keys=True))
        if key in seen:
            found.append("repeated_call")
            break
        seen.add(key)
    path = handoff_path(case, trace)
    if len(path) != len(set(path)):
        found.append("handoff_returned")
    return found
