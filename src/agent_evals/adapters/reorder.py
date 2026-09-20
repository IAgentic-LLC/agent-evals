"""Adapters for the reorder product's approval workflow (chapter 11).

The workflow is a conversation with two turns. The first turn asks a question, and
the agent checks stock and either answers or pauses for a human. The second turn is
the human's decision, which resumes the paused run. The state between them is saved
under a thread id in a checkpointer. Each run here gets its own thread and its own
database file, so runs cannot see each other, as chapter 7 requires.

  ReorderAdapter          the real model
  ReorderScriptedAdapter  a scripted model, so the mechanics can be checked with no
                          key and no cost
"""

import os

# The workflow's functions are traced with Langfuse. These placeholder settings turn
# tracing off, so nothing is sent anywhere and no warning is printed. They are only
# set when nothing else has set them.
os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
os.environ.setdefault("LANGFUSE_PUBLIC_KEY", "pk-lf-unused")
os.environ.setdefault("LANGFUSE_SECRET_KEY", "sk-lf-unused")

import json
import re
import tempfile
import time
from pathlib import Path
from typing import Any

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from reliable_agents_labs.inventory import check_inventory
from reliable_agents_labs.models import ModelResult, ToolCall
from reliable_agents_labs.reorder_workflow import build_approval_workflow

from agent_evals.recording import RecordingClient
from agent_evals.schema import EvalCase, Trace

# The state a new request starts from. It is the same dictionary the product's own
# API builds in `reorder_app.api._EMPTY_APPROVAL_STATE`, copied here because importing
# that module starts the web app.
EMPTY_STATE = {
    "question": "",
    "answer": "",
    "reorder": False,
    "approved": False,
    "note": "",
    "logged": False,
}
SKU = re.compile(r"SKU-\d+")


class ScriptedReorderClient:
    """A stand-in model: it looks up the first SKU in the question and reports what
    the inventory says, then answers the workflow's follow-up decision honestly."""

    def __init__(self) -> None:
        self.calls = 0

    @staticmethod
    def _result(text: str, calls: list[ToolCall] | None = None) -> ModelResult:
        return ModelResult(
            text=text,
            input_tokens=1,
            output_tokens=1,
            model_id="scripted",
            provider="scripted",
            tool_calls=calls or [],
        )

    async def generate(self, *, system, user, tools=None, history=None):
        self.calls += 1
        if not tools:  # the workflow's decision call: does the answer say reorder?
            return self._result(json.dumps({"reorder": "should reorder" in user}))
        if not history:  # first turn of the tool loop: look the SKU up
            match = SKU.search(user)
            sku = match.group(0) if match else "SKU-0000"
            call = ToolCall(id="c1", name="check_inventory", arguments={"sku": sku})
            return self._result("", [call])
        record = json.loads(history[-1]["content"])
        low = "error" not in record and record["quantity"] < record["reorder_point"]
        return self._result(
            "You should reorder this SKU." if low else "There is no need to reorder."
        )


def scripted_stock_says_reorder(question: str) -> bool:
    """What the scripted model will conclude for a question (first SKU only)."""
    match = SKU.search(question)
    record = check_inventory(match.group(0)) if match else None
    return record is not None and record.quantity < record.reorder_point


async def converse(
    graph, recorder: RecordingClient, case: EvalCase, thread_id: str
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Play a case's turns against a graph and record what each turn did."""
    config = {"configurable": {"thread_id": thread_id}}
    turns: list[dict[str, Any]] = []

    async def observe(sent: str, result: dict[str, Any], calls_before: int) -> None:
        snapshot = await graph.aget_state(config)
        turns.append(
            {
                "turn": len(turns) + 1,
                "sent": sent,
                "paused": "__interrupt__" in result,
                "next": list(snapshot.next),
                "model_calls": recorder.model_calls - calls_before,
                "reorder": bool(snapshot.values.get("reorder")),
                "logged": bool(snapshot.values.get("logged")),
            }
        )

    before = recorder.model_calls
    state = {**EMPTY_STATE, "question": case.input["question"]}
    await observe("ask", await graph.ainvoke(state, config), before)
    decision = case.input.get("decision")
    if turns[-1]["paused"] and decision in ("approve", "reject"):
        before = recorder.model_calls
        resume = {
            "approved": decision == "approve",
            "note": f"{decision} from the eval",
        }
        await observe(
            decision, await graph.ainvoke(Command(resume=resume), config), before
        )
    final = await graph.aget_state(config)
    return turns, dict(final.values)


class ReorderAdapter:
    """The approval workflow with the real model, one conversation per case."""

    name = "reorder-live"

    def __init__(self, client=None) -> None:
        self._client = client

    def _model(self):
        if self._client is not None:
            return self._client
        from reliable_agents_labs.models import build_model_client

        return build_model_client("answer_model")

    async def run(self, case: EvalCase, trial: int) -> Trace:
        started = time.perf_counter()
        turns: list[dict[str, Any]] = []
        values: dict[str, Any] = {}
        error = None
        recorder = RecordingClient(self._model())
        with tempfile.TemporaryDirectory() as folder:
            db = str(Path(folder) / "checkpoints.sqlite")
            async with AsyncSqliteSaver.from_conn_string(db) as saver:
                graph = build_approval_workflow(
                    model_client=recorder, checkpointer=saver
                )
                try:
                    thread = f"{case.case_id}-t{trial}"
                    turns, values = await converse(graph, recorder, case, thread)
                except Exception as exc:  # noqa: BLE001 - a failed run is a result
                    error = f"{type(exc).__name__}: {exc}"
        acted = [{"action": "log_reorder", "question": case.input["question"]}]
        return Trace(
            case_id=case.case_id,
            trial=trial,
            adapter=self.name,
            answer=values.get("answer", ""),
            actions_taken=acted if values.get("logged") else [],
            error=error,
            latency_s=round(time.perf_counter() - started, 3),
            tool_calls=recorder.finish(),
            turns=turns,
        )


class ReorderScriptedAdapter(ReorderAdapter):
    name = "reorder-scripted"

    def _model(self):
        return ScriptedReorderClient()
