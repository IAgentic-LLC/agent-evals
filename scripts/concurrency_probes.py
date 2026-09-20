"""Two things that go wrong when evaluation runs overlap (chapter 3).

1. The product records the actions its tools take in one context-local list.
   If the caller has touched that list before the runs start, every run shares it.
2. An adapter that patches the product for each run, and restores it after,
   leaves the product patched when the runs overlap.

Usage: uv run python scripts/concurrency_probes.py
"""

import asyncio

from triage_app import specialists
from triage_app.tools import (
    ACTIONS_TAKEN,
    BILLING_TOOLS,
    escalate_to_oncall,
    look_up_invoice,
    search_runbook,
)

from agent_evals.adapters.triage import RegressedAdapter
from agent_evals.runner import load_cases, run_cases


async def _work(name, tool, args):
    ACTIONS_TAKEN.clear()  # how the harness reset the recorder before this chapter
    await asyncio.sleep(0.01)  # a model call: other runs proceed meanwhile
    tool(args)
    await asyncio.sleep(0.01)
    return name, [a["action"] for a in ACTIONS_TAKEN]


async def shared_recorder(touch_first: bool) -> list[tuple[str, list[str]]]:
    if touch_first:
        ACTIONS_TAKEN.clear()  # something in the caller used the recorder first
    runs = [
        _work("billing", look_up_invoice, {"customer_id": "cust-42"}),
        _work("technical", search_runbook, {"query": "app crash"}),
        _work("security", escalate_to_oncall, {"reason": "unrecognized login"}),
    ]
    return list(await asyncio.gather(*runs))


class PatchEachRunAdapter(RegressedAdapter):
    """The first version of the regression adapter: patch and restore per run."""

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    async def run(self, case, trial):
        refund = next(
            t for t in BILLING_TOOLS if t["function"]["name"] == "issue_refund"
        )
        original = specialists.TECHNICAL_TOOLS
        specialists.TECHNICAL_TOOLS = original + [refund]
        try:
            return await super().run(case, trial)
        finally:
            specialists.TECHNICAL_TOOLS = original


async def patch_state(adapter, concurrency: int) -> tuple[int, int]:
    """How many tools the technical specialist has before and after a run."""
    cases = load_cases("datasets/triage_book3_six.jsonl")
    before = len(specialists.TECHNICAL_TOOLS)
    await run_cases(cases, adapter, concurrency=concurrency)
    return before, len(specialists.TECHNICAL_TOOLS)


def main() -> None:
    for touch in (False, True):
        print(f"caller touched the recorder first: {touch}")
        for name, actions in asyncio.run(shared_recorder(touch)):
            print(f"  {name:<10} {actions}")
    print()
    for label, adapter, workers in (
        ("patch per run, one at a time", PatchEachRunAdapter(delay_s=0.02), 1),
        ("patch per run, six at a time", PatchEachRunAdapter(delay_s=0.02), 6),
    ):
        before, after = asyncio.run(patch_state(adapter, workers))
        print(f"{label}: technical tools {before} before, {after} after")
    specialists.TECHNICAL_TOOLS = specialists.TECHNICAL_TOOLS[:2]


if __name__ == "__main__":
    main()
