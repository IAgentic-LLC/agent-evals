"""Adapters that let the harness evaluate the triage-app product from Book 3.

Four ways to produce a trace for the same case:
  LiveAdapter        the real model (needs a GEMINI_API_KEY in the environment)
  ReplayAdapter      traces recorded from an earlier live run, so anyone can reproduce
                     the scores without a key
  ScriptedWallAdapter    a scripted model against the real topology, where each
                     specialist only has its own tools
  RegressedAdapter   the same scripted model after the technical specialist has been
                     "helpfully" given the refund tool: a deliberate regression, used
                     to show which gates can catch it
"""

import time
from pathlib import Path

from reliable_agents_labs.models import ModelResult, ToolCall
from triage_app.supervisor import route_ticket
from triage_app.tickets import Ticket
from triage_app.tools import ACTIONS_TAKEN, BILLING_TOOLS

from agent_evals.runner import read_traces
from agent_evals.schema import EvalCase, Trace


def _ticket(case: EvalCase) -> Ticket:
    return Ticket(**case.input)


def _result(text: str, calls: list[ToolCall] | None = None) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=10,
        output_tokens=5,
        model_id="scripted",
        provider="scripted",
        tool_calls=calls or [],
    )


class _ScriptedClient:
    def __init__(self, results: list[ModelResult]) -> None:
        self._results = iter(results)

    async def generate(self, *, system, user, tools=None, history=None) -> ModelResult:
        return next(self._results)


def _script_for(case: EvalCase) -> list[ModelResult]:
    """A model that follows any injection in the ticket and otherwise behaves."""
    category = case.input["category"]
    expected = case.expected["handled_by"]
    if case.invariants.get("forbidden_actions"):
        refund = ToolCall(
            id="c1",
            name="issue_refund",
            arguments={"customer_id": case.input["customer_id"], "amount_usd": 999},
        )
        return [_result("", [refund]), _result("Refund issued as requested.")]
    if expected != category:
        handoff = ToolCall(
            id="c1",
            name="request_handoff",
            arguments={"target_category": expected, "reason": "Not in my domain."},
        )
        return [_result("", [handoff]), _result("Handled after the handoff.")]
    return [_result("Resolved.")]


async def _run_once(case: EvalCase, trial: int, adapter: str, client) -> Trace:
    ACTIONS_TAKEN.clear()
    started = time.perf_counter()
    handled_by, error = None, None
    try:
        resolution = await route_ticket(_ticket(case), client=client)
        handled_by = resolution.handled_by
    except Exception as exc:  # noqa: BLE001 - a failed run is a result, not a crash of the harness
        error = f"{type(exc).__name__}: {exc}"
    return Trace(
        case_id=case.case_id,
        trial=trial,
        adapter=adapter,
        handled_by=handled_by,
        actions_taken=[dict(a) for a in ACTIONS_TAKEN],
        error=error,
        latency_s=round(time.perf_counter() - started, 3),
    )


class LiveAdapter:
    name = "triage-live"

    async def run(self, case: EvalCase, trial: int) -> Trace:
        return await _run_once(case, trial, self.name, client=None)


class ScriptedWallAdapter:
    name = "triage-scripted-wall"

    async def run(self, case: EvalCase, trial: int) -> Trace:
        return await _run_once(
            case, trial, self.name, _ScriptedClient(_script_for(case))
        )


class RegressedAdapter:
    name = "triage-regressed-scripted"

    async def run(self, case: EvalCase, trial: int) -> Trace:
        from triage_app import specialists

        refund_tool = next(
            t for t in BILLING_TOOLS if t["function"]["name"] == "issue_refund"
        )
        original = specialists.TECHNICAL_TOOLS
        specialists.TECHNICAL_TOOLS = original + [refund_tool]
        try:
            return await _run_once(
                case, trial, self.name, _ScriptedClient(_script_for(case))
            )
        finally:
            specialists.TECHNICAL_TOOLS = original


class ReplayAdapter:
    name = "triage-replay"

    def __init__(self, traces_path: str | Path) -> None:
        self._by_key = {(t.case_id, t.trial): t for t in read_traces(traces_path)}

    async def run(self, case: EvalCase, trial: int) -> Trace:
        recorded = self._by_key[(case.case_id, trial)]
        return recorded.model_copy(update={"note": "replayed from a recorded run"})
