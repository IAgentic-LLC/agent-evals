"""Chapter 3: the harness has to be trustworthy before its numbers are.

No API key needed: scripted models with a delay stand in for model latency.
"""

import asyncio
import hashlib
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

from triage_app import specialists
from triage_app.tools import ACTIONS_TAKEN

from agent_evals.adapters.triage import (
    CustomerIdAdapter,
    RegressedAdapter,
    ScriptedWallAdapter,
)
from agent_evals.manifest import build_manifest
from agent_evals.runner import load_cases, run_cases
from agent_evals.schema import EvalCase, Trace
from agent_evals.scorecard import build_scorecard, render_markdown

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "triage_book3_six.jsonl"


def _probes():
    spec = importlib.util.spec_from_file_location(
        "concurrency_probes", ROOT / "scripts" / "concurrency_probes.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _shape(traces):
    return [
        (
            t.case_id,
            t.trial,
            t.handled_by,
            [a["action"] for a in t.actions_taken],
            t.error,
        )
        for t in traces
    ]


async def test_the_shared_recorder_mixes_runs_up_when_the_caller_touched_it_first():
    probes = _probes()
    clean = await probes.shared_recorder(touch_first=False)
    touched = await probes.shared_recorder(touch_first=True)
    assert [len(a) for _, a in clean] == [1, 1, 1]
    assert [len(a) for _, a in touched] == [3, 3, 3]


async def test_overlapping_runs_score_the_same_as_one_at_a_time_even_if_the_caller_touched_the_recorder():
    cases = load_cases(DATASET)
    one_at_a_time = await run_cases(cases, ScriptedWallAdapter(delay_s=0.01), trials=2)
    ACTIONS_TAKEN.clear()  # the caller touched the recorder before the runs
    overlapping = await run_cases(
        cases, ScriptedWallAdapter(delay_s=0.01), trials=2, concurrency=6
    )
    assert _shape(overlapping) == _shape(one_at_a_time)


async def test_traces_come_back_in_case_then_trial_order_whatever_finishes_first():
    class Staggered:
        name = "staggered"

        async def run(self, case: EvalCase, trial: int) -> Trace:
            await asyncio.sleep(0.03 if case.case_id == "TCK-1001" else 0.0)
            return Trace(case_id=case.case_id, trial=trial, adapter=self.name)

    cases = load_cases(DATASET)
    traces = await run_cases(cases, Staggered(), trials=2, concurrency=12)
    expected = [(c.case_id, t) for c in cases for t in (1, 2)]
    assert [(t.case_id, t.trial) for t in traces] == expected


async def test_a_concurrent_regression_run_leaves_the_product_as_it_found_it():
    before = list(specialists.TECHNICAL_TOOLS)
    original_question = specialists._question_for
    cases = load_cases(DATASET)
    await run_cases(cases, RegressedAdapter(delay_s=0.01), concurrency=6)
    assert specialists.TECHNICAL_TOOLS == before
    assert specialists._question_for is original_question


async def test_patching_per_run_leaves_the_product_altered_when_runs_overlap():
    probes = _probes()
    try:
        _, one_at_a_time = await probes.patch_state(
            probes.PatchEachRunAdapter(delay_s=0.01), concurrency=1
        )
        _, overlapping = await probes.patch_state(
            probes.PatchEachRunAdapter(delay_s=0.01), concurrency=6
        )
    finally:
        specialists.TECHNICAL_TOOLS = specialists.TECHNICAL_TOOLS[:2]
    assert one_at_a_time == 2
    assert overlapping > 2


def test_the_manifest_ties_a_run_to_its_dataset_and_settings():
    manifest = build_manifest(
        harness={"commit": "abc", "dirty": False},
        adapter="triage-live",
        dataset=DATASET,
        cases=6,
        trials=3,
        concurrency=6,
        started_at=datetime(2026, 9, 20, 12, 0, tzinfo=UTC),
        wall_seconds=12.345,
    )
    assert (
        manifest["dataset"]["sha256"]
        == hashlib.sha256(DATASET.read_bytes()).hexdigest()
    )
    assert manifest["trials"] == 3 and manifest["concurrency"] == 6
    assert manifest["started_at"] == "2026-09-20T12:00:00+00:00"
    assert manifest["wall_seconds"] == 12.35
    # The installed product is the tag the book pins, identified by its commit.
    triage = manifest["packages"]["triage-app"]
    assert triage["requested_revision"] == "ch35-end"
    assert len(triage["commit"]) == 40
    assert manifest["harness"] == {"commit": "abc", "dirty": False}


async def test_a_scorecard_from_repeated_trials_says_its_intervals_describe_the_trials():
    cases = load_cases(DATASET)
    once = build_scorecard(
        "six", "r", cases, await run_cases(cases, ScriptedWallAdapter())
    )
    thrice = build_scorecard(
        "six", "r", cases, await run_cases(cases, ScriptedWallAdapter(), trials=3)
    )
    assert "not independent" not in render_markdown(once)
    assert "not independent" in render_markdown(thrice)


class _CapturingClient:
    def __init__(self) -> None:
        self.questions: list[str] = []

    async def generate(self, *, system, user, tools=None, history=None):
        from reliable_agents_labs.models import ModelResult

        await asyncio.sleep(0.01)
        self.questions.append(user)
        return ModelResult(
            text="ok",
            input_tokens=1,
            output_tokens=1,
            model_id="scripted",
            provider="scripted",
            tool_calls=[],
        )


async def test_the_customer_id_patch_applies_once_however_many_runs_overlap():
    original_question = specialists._question_for
    client = _CapturingClient()
    cases = load_cases(DATASET)
    await run_cases(cases, CustomerIdAdapter(client=client), trials=2, concurrency=12)
    assert len(client.questions) == 12
    assert all(q.count("Customer ID:") == 1 for q in client.questions)
    assert specialists._question_for is original_question


async def test_a_crashing_adapter_cancels_the_other_runs_and_restores_the_product():
    before = list(specialists.TECHNICAL_TOOLS)
    finished: list[str] = []

    class Crashing(RegressedAdapter):
        async def run(self, case: EvalCase, trial: int) -> Trace:
            if case.case_id == "TCK-1002":
                raise RuntimeError("adapter bug")
            await asyncio.sleep(0.2)
            finished.append(case.case_id)
            return await super().run(case, trial)

    cases = load_cases(DATASET)
    try:
        await run_cases(cases, Crashing(), concurrency=6)
    except* RuntimeError:
        pass
    else:
        raise AssertionError("the adapter bug should have surfaced")
    assert finished == []  # the other runs were cancelled, not left running
    assert specialists.TECHNICAL_TOOLS == before
