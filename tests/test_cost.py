"""Chapter 22: what a run costs, how long it takes, and what an evaluation costs."""

import pytest
from reliable_agents_labs.models import ModelResult, ToolCall

from agent_evals.recording import RecordingClient
from agent_evals.schema import Trace


def _result(text, calls=None, tokens_in=10, tokens_out=5):
    return ModelResult(
        text=text,
        input_tokens=tokens_in,
        output_tokens=tokens_out,
        model_id="scripted",
        provider="scripted",
        tool_calls=calls or [],
    )


class _Inner:
    def __init__(self, replies):
        self._replies = iter(replies)

    async def generate(self, *, system, user, tools=None, history=None):
        return next(self._replies)


async def test_the_recorder_sums_the_calls_and_tokens_the_provider_reported():
    call = ToolCall(id="c1", name="search_runbook", arguments={"query": "q"})
    inner = _Inner([_result("", [call], 100, 20), _result("Done.", None, 150, 30)])
    recorder = RecordingClient(inner)
    await recorder.generate(system="s", user="u", tools=None)
    await recorder.generate(system="s", user="u", tools=None)
    assert recorder.usage() == {
        "model_calls": 2,
        "input_tokens": 250,
        "output_tokens": 50,
        "thinking_tokens": 0,
    }


def test_a_trace_from_before_chapter_22_has_an_empty_usage():
    assert Trace(case_id="X", adapter="a").usage == {}


def test_the_manifest_records_the_model_a_run_was_told_to_use():
    from datetime import UTC, datetime

    from agent_evals.manifest import build_manifest

    manifest = build_manifest(
        harness=None,
        adapter="triage-live-customer-id",
        dataset="datasets/triage_heldout_v1.jsonl",
        cases=42,
        trials=1,
        concurrency=1,
        started_at=datetime.now(UTC),
        wall_seconds=1.0,
        model={"provider": "gemini", "model_id": "gemini-2.5-flash"},
    )
    assert manifest["model"]["model_id"] == "gemini-2.5-flash"
    assert pytest.approx(1.0) == manifest["wall_seconds"]
