"""Chapter 7: each run needs its own state, and a leak must be detectable.

No API key needed: a scripted model, the recorded runs, and the demo functions.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
from reliable_agents_labs.models import ModelResult, ToolCall
from triage_app.tools import _INVOICES

from agent_evals import world
from agent_evals.adapters.triage import _run_once
from agent_evals.graders import forbidden_actions_taken, required_actions_missing
from agent_evals.runner import load_cases, read_traces
from agent_evals.scorecard import build_scorecard

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")}
LEAKY = ROOT / "runs" / "triage-heldout-v1-leaky" / "traces.jsonl"


class _SearchThenAnswer:
    """A scripted model that searches the runbook once, then answers."""

    def __init__(self):
        self._calls = 0

    async def generate(self, *, system, user, tools=None, history=None):
        self._calls += 1
        if self._calls == 1:
            call = ToolCall(
                id="c1", name="search_runbook", arguments={"query": "app crash"}
            )
            return _result("", [call])
        return _result("Done.")


def _result(text, calls=None):
    return ModelResult(
        text=text,
        input_tokens=1,
        output_tokens=1,
        model_id="scripted",
        provider="scripted",
        tool_calls=calls or [],
    )


async def test_a_run_without_the_reset_inherits_the_last_runs_actions():
    case = CASES["HO-011"]  # a technical ticket
    first = await _run_once(case, 1, "t", _SearchThenAnswer())
    second = await _run_once(case, 1, "t", _SearchThenAnswer(), isolate=False)
    assert first.ledger_at_start == 0 and len(first.actions_taken) == 1
    assert second.ledger_at_start == 1 and len(second.actions_taken) == 2


async def test_a_run_with_the_reset_starts_clean_however_dirty_the_ledger_was():
    case = CASES["HO-011"]
    await _run_once(case, 1, "t", _SearchThenAnswer())
    again = await _run_once(case, 1, "t", _SearchThenAnswer())
    assert again.ledger_at_start == 0 and len(again.actions_taken) == 1


def test_the_state_rules_flag_a_run_that_started_with_leftovers():
    case = CASES["HO-011"]
    clean = read_traces(ROOT / "runs/triage-heldout-v1/traces.jsonl")[0]
    dirty = clean.model_copy(update={"ledger_at_start": 3})
    assert "started_with_leftover_state" not in world.state_violations(
        case, clean, _INVOICES
    )
    assert "started_with_leftover_state" in world.state_violations(
        case, dirty, _INVOICES
    )


def test_the_recorded_leaky_run_started_dirty_almost_every_time():
    traces = read_traces(LEAKY)
    assert len(traces) == 42
    assert [t.ledger_at_start for t in traces][:4] == [0, 2, 3, 4]
    assert sum(t.ledger_at_start > 0 for t in traces) == 41
    counts = {}
    for t in traces:
        for rule in world.state_violations(CASES[t.case_id], t, _INVOICES):
            counts[rule] = counts.get(rule, 0) + 1
    assert counts["started_with_leftover_state"] == 41
    assert counts["acted_on_another_customer"] == 39


def _own(trace):
    """The trace with only the actions this run took itself."""
    return trace.model_copy(
        update={"actions_taken": trace.actions_taken[trace.ledger_at_start :]}
    )


def test_the_leak_added_two_passes_and_five_false_alarms_and_nothing_else():
    traces = read_traces(LEAKY)
    as_scored = own_only = 0
    inherited_passes, false_alarms = [], []
    for t in traces:
        case = CASES[t.case_id]
        scored = t.error is None and not required_actions_missing(case, t)
        own = t.error is None and not required_actions_missing(case, _own(t))
        as_scored += scored
        own_only += own
        if scored and not own:
            inherited_passes.append(t.case_id)
        if forbidden_actions_taken(case, t) and not forbidden_actions_taken(
            case, _own(t)
        ):
            false_alarms.append(t.case_id)
    assert (as_scored, own_only) == (31, 29)
    assert inherited_passes == ["HO-026", "HO-041"]
    assert false_alarms == ["HO-037", "HO-038", "HO-039", "HO-041", "HO-042"]


def test_clean_runs_of_the_same_product_vary_about_as_much_as_the_leak_moved_the_score():
    passes = []
    for name in ("triage-heldout-v1", "triage-heldout-v1-2", "triage-heldout-v1-3"):
        traces = read_traces(ROOT / "runs" / name / "traces.jsonl")
        card = build_scorecard("t", name, list(CASES.values()), traces)
        passes.append(card.actions_successes)
    assert passes == [26, 25, 28]
    leaky = build_scorecard("t", "leaky", list(CASES.values()), read_traces(LEAKY))
    assert leaky.actions_successes == 31 and leaky.invariant_violations == 5


def test_the_demo_functions_show_isolation(capsys):
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("demos", ROOT / "scripts" / "sandbox_demos.py")
    demos = module_from_spec(spec)
    spec.loader.exec_module(demos)
    demos.temp_directory_per_run()
    demos.controlled_clock()
    demos.database_per_run()
    out = capsys.readouterr().out
    assert "run B sees:  []" in out and "still exists: False" in out
    assert "same output twice: True" in out
    assert "run A sees 1 refund" in out and "run B sees 0 refunds" in out


def _docker_ready() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        return (
            subprocess.run(
                ["docker", "info"], capture_output=True, timeout=20, check=False
            ).returncode
            == 0
        )
    except (OSError, subprocess.TimeoutExpired):
        return False


@pytest.mark.skipif(not _docker_ready(), reason="Docker is not running")
def test_a_throwaway_postgres_container_per_run_isolates_state(capsys):
    pytest.importorskip("testcontainers")
    from importlib.util import module_from_spec, spec_from_file_location

    spec = spec_from_file_location("demos", ROOT / "scripts" / "sandbox_demos.py")
    demos = module_from_spec(spec)
    spec.loader.exec_module(demos)
    demos.throwaway_postgres()
    out = capsys.readouterr().out
    assert "run A sees 1 refund" in out and "run B sees 0 refunds" in out
