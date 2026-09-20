"""Checks on the mechanics of a workflow that pauses for a human and resumes (chapter 11).

These use a scripted model, so they need no key and give the same answer every time.
Each check plays one small conversation against a workflow and reports what went wrong,
or nothing. The real workflow should pass all of them. The other variants each have one
deliberate fault, to show that a check can fail: a detector that has never caught a
known fault is one you have not tested.
"""

import tempfile
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from reliable_agents_labs import reorder_workflow as flow

from agent_evals.adapters.reorder import EMPTY_STATE, ScriptedReorderClient

REORDER_LOG = flow.REORDER_LOG


def _real(saver: Any, client: Any) -> Any:
    return flow.build_approval_workflow(model_client=client, checkpointer=saver)


def _acts_before_approval(saver: Any, client: Any) -> Any:
    """The reorder is logged as soon as the agent recommends it. There is no pause."""
    builder = StateGraph(flow.ApprovalWorkflowState)
    builder.add_node("ask_agent", flow._build_ask_agent_node(client))
    builder.add_node("log_reorder", flow.log_reorder)
    builder.add_edge(START, "ask_agent")
    routes = {"log_reorder": "log_reorder", "skip": END}
    builder.add_conditional_edges("ask_agent", flow._route_on_answer, routes)
    builder.add_edge("log_reorder", END)
    return builder.compile(checkpointer=saver)


def _pause_inside_the_model_node(saver: Any, client: Any) -> Any:
    """The pause is in the same node as the model call. On resume the node runs again
    from its start, so the model is asked again."""
    ask_agent = flow._build_ask_agent_node(client)

    async def ask_then_wait(state: dict) -> dict:
        update = await ask_agent(state)
        if not update["reorder"]:
            return update
        decision = interrupt(
            {"question": state["question"], "answer": update["answer"]}
        )
        return {
            **update,
            "approved": decision["approved"],
            "note": decision.get("note", ""),
        }

    def route(state: dict) -> str:
        return "log_reorder" if state["reorder"] and state["approved"] else "skip"

    builder = StateGraph(flow.ApprovalWorkflowState)
    builder.add_node("ask_then_wait", ask_then_wait)
    builder.add_node("log_reorder", flow.log_reorder)
    builder.add_edge(START, "ask_then_wait")
    builder.add_conditional_edges(
        "ask_then_wait", route, {"log_reorder": "log_reorder", "skip": END}
    )
    builder.add_edge("log_reorder", END)
    return builder.compile(checkpointer=saver)


def _logs_twice(saver: Any, client: Any) -> Any:
    """The same workflow, with the action written twice."""

    def log_twice(state: dict) -> dict:
        flow.log_reorder(state)
        return flow.log_reorder(state)

    builder = StateGraph(flow.ApprovalWorkflowState)
    builder.add_node("ask_agent", flow._build_ask_agent_node(client))
    builder.add_node("await_approval", flow.await_approval)
    builder.add_node("log_reorder", log_twice)
    builder.add_edge(START, "ask_agent")
    builder.add_conditional_edges(
        "ask_agent",
        flow._route_on_answer,
        {"log_reorder": "await_approval", "skip": END},
    )
    builder.add_conditional_edges(
        "await_approval",
        flow._route_on_approval,
        {"log_reorder": "log_reorder", "skip": END},
    )
    builder.add_edge("log_reorder", END)
    return builder.compile(checkpointer=saver)


def _sqlite(path: str):
    return AsyncSqliteSaver.from_conn_string(path)


@asynccontextmanager
async def _forgetful(path: str):
    """A checkpointer that keeps state only while the process is up."""
    yield InMemorySaver()


@dataclass(frozen=True)
class Variant:
    name: str
    build: Callable[[Any, Any], Any]
    saver: Callable[[str], Any] = _sqlite
    one_thread_for_all: bool = False


VARIANTS = {
    v.name: v
    for v in (
        Variant("the real workflow", _real),
        Variant("acts before approval", _acts_before_approval),
        Variant("pauses inside the model node", _pause_inside_the_model_node),
        Variant("logs the action twice", _logs_twice),
        Variant("forgets on restart", _real, saver=_forgetful),
        Variant("one thread for everyone", _real, one_thread_for_all=True),
    )
}


def _ask(question: str) -> dict:
    return {**EMPTY_STATE, "question": question}


def _config(thread: str) -> dict:
    return {"configurable": {"thread_id": thread}}


def _approve(approved: bool) -> Command:
    return Command(resume={"approved": approved, "note": "from the check"})


LOW = "Do we need to reorder SKU-1029?"
FINE = "Do we need to reorder SKU-2040?"
OTHER_LOW = "Do we need to reorder SKU-3311?"


async def pauses_before_acting(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph, before = v.build(saver, ScriptedReorderClient()), len(REORDER_LOG)
        result = await graph.ainvoke(_ask(LOW), _config("t"))
    if len(REORDER_LOG) != before:
        return "logged an order before anyone approved it"
    return None if "__interrupt__" in result else "did not pause for a decision"


async def does_not_pause_when_not_needed(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph = v.build(saver, ScriptedReorderClient())
        result = await graph.ainvoke(_ask(FINE), _config("t"))
    return (
        "paused although no reorder was needed" if "__interrupt__" in result else None
    )


async def resume_asks_the_model_nothing(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        client = ScriptedReorderClient()
        graph = v.build(saver, client)
        result = await graph.ainvoke(_ask(LOW), _config("t"))
        if "__interrupt__" not in result:
            return "did not pause, so there was nothing to resume"
        asked = client.calls
        await graph.ainvoke(_approve(True), _config("t"))
    return (
        None
        if client.calls == asked
        else f"asked the model {client.calls - asked} more times"
    )


async def approval_acts_once(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph = v.build(saver, ScriptedReorderClient())
        result = await graph.ainvoke(_ask(LOW), _config("t"))
        before = len(REORDER_LOG)
        if "__interrupt__" not in result:
            return "did not pause, so there was nothing to approve"
        await graph.ainvoke(_approve(True), _config("t"))
    acted = len(REORDER_LOG) - before
    return None if acted == 1 else f"acted {acted} times after approval"


async def rejection_does_not_act(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph, before = v.build(saver, ScriptedReorderClient()), len(REORDER_LOG)
        result = await graph.ainvoke(_ask(LOW), _config("t"))
        if "__interrupt__" in result:
            await graph.ainvoke(_approve(False), _config("t"))
    if len(REORDER_LOG) != before:
        return "acted although no one had approved it"
    return None


async def state_survives_a_restart(v: Variant, folder: str) -> str | None:
    db, before = f"{folder}/a.sqlite", len(REORDER_LOG)
    async with v.saver(db) as saver:
        result = await v.build(saver, ScriptedReorderClient()).ainvoke(
            _ask(LOW), _config("t")
        )
    if "__interrupt__" not in result:
        return "did not pause, so there was nothing to resume"
    # A new process: a new saver on the same file, a new graph, a new model client.
    async with v.saver(db) as saver:
        await v.build(saver, ScriptedReorderClient()).ainvoke(
            _approve(True), _config("t")
        )
    entries = REORDER_LOG[before:]
    if not entries:
        return "lost the paused run"
    if len(entries) != 1 or "SKU-1029" not in entries[0]:
        return f"expected one entry for SKU-1029, got {len(entries)}"
    return None


async def a_second_approval_acts_once(v: Variant, folder: str) -> str | None:
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph = v.build(saver, ScriptedReorderClient())
        result = await graph.ainvoke(_ask(LOW), _config("t"))
        before = len(REORDER_LOG)
        if "__interrupt__" not in result:
            return "did not pause, so there was nothing to approve"
        await graph.ainvoke(_approve(True), _config("t"))
        await graph.ainvoke(_approve(True), _config("t"))
    acted = len(REORDER_LOG) - before
    return None if acted == 1 else f"acted {acted} times after two approvals"


async def threads_do_not_mix(v: Variant, folder: str) -> str | None:
    a, b = ("t", "t") if v.one_thread_for_all else ("a", "b")
    async with v.saver(f"{folder}/a.sqlite") as saver:
        graph, before = v.build(saver, ScriptedReorderClient()), len(REORDER_LOG)
        await graph.ainvoke(_ask(LOW), _config(a))
        await graph.ainvoke(_ask(OTHER_LOW), _config(b))
        await graph.ainvoke(_approve(False), _config(b))
        await graph.ainvoke(_approve(True), _config(a))
    entries = REORDER_LOG[before:]
    if len(entries) == 1 and "SKU-1029" in entries[0]:
        return None
    return f"expected one entry, for SKU-1029, got {len(entries)}"


CHECKS: dict[str, Callable[[Variant, str], Any]] = {
    fn.__name__: fn
    for fn in (
        pauses_before_acting,
        does_not_pause_when_not_needed,
        resume_asks_the_model_nothing,
        approval_acts_once,
        rejection_does_not_act,
        state_survives_a_restart,
        a_second_approval_acts_once,
        threads_do_not_mix,
    )
}


async def run_mechanics(variant: Variant) -> dict[str, str | None]:
    """Every check against one variant: the failure text, or None for a pass."""
    found: dict[str, str | None] = {}
    for name, check in CHECKS.items():
        with tempfile.TemporaryDirectory() as folder:
            Path(folder).mkdir(exist_ok=True)
            try:
                found[name] = await check(variant, folder)
            except Exception as exc:  # noqa: BLE001 - an exception is a failed check
                found[name] = f"raised {type(exc).__name__}: {str(exc)[:60]}"
    return found
