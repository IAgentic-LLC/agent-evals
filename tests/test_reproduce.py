"""Appendix A: the command that made a recorded run, and what it cost.

The tool reads manifests and traces and never calls a model, so every test runs offline.
"""

import contextlib
import io
import json
from pathlib import Path

from agent_evals import cli, cost, reproduce
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
PRICES = cost.load_prices(ROOT / "config/prices.yaml")


def _cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["reproduce", *args]) == 0
    text = buf.getvalue()
    assert all(len(line) <= 78 for line in text.splitlines())
    return text


def test_a_recorded_run_becomes_the_command_that_made_it():
    text = _cli("--part", "command", "--run", "triage-metered-3-6")
    assert "--adapter triage-live-customer-id" in text
    assert "--dataset datasets/triage_heldout_v1.jsonl" in text
    assert "--trials 3" in text and "--concurrency 6" in text
    assert "--answer-model gemini-3.6-flash" in text
    assert "--out runs/mine-metered-3-6" in text
    assert text.count(chr(92)) == 6 and "--env-file .env" in text


def test_every_flag_in_the_command_is_one_the_run_command_accepts():
    text = _cli("--part", "command", "--run", "triage-heldout-lite-idboth")
    flags = {w for w in text.split() if w.startswith("--")}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.suppress(SystemExit):
        cli.main(["run", "--help"])
    accepted = buf.getvalue()
    assert flags and all(flag in accepted for flag in flags)
    assert "--answer-model gemini-3.5-flash-lite" in text.replace(chr(92), " ").replace(
        chr(10), " "
    ).replace("  ", " ")


def test_the_cost_is_the_recorded_tokens_at_the_price_file_and_is_called_a_floor():
    data = reproduce.manifest(ROOT, "triage-metered-3-6")
    dollars, runs = reproduce.estimate(ROOT, "triage-metered-3-6", data, PRICES)
    assert runs == 126 and 0.45 < dollars < 0.55
    text = _cli("--part", "command", "--run", "triage-metered-3-6")
    assert "about $0.50" in text and "floor" in text


def test_the_list_shows_one_row_per_adapter_run_with_its_cost_and_whether_it_needs_a_key():
    text = _cli("--part", "list", "--match", "metered")
    lines = text.splitlines()
    assert lines[0].split()[0] == "run" and lines[-1].startswith("6 runs")
    row = next(ln for ln in lines if ln.startswith("triage-metered-3-5-lite"))
    assert row.split()[1:] == ["3.5-flash-lite", "3", "126", "$0.09", "yes"]


def test_a_run_that_needs_no_key_says_so(tmp_path):
    run = tmp_path / "runs" / "scripted"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "adapter": "triage-scripted-wall",
                "dataset": {"path": "datasets/x.jsonl"},
                "trials": 1,
                "concurrency": 1,
            }
        ),
        encoding="utf8",
    )
    text = reproduce.render_command(tmp_path, "scripted", PRICES, ())
    assert "This adapter needs no key." in text and "--env-file" not in text


def test_a_run_without_a_manifest_or_from_another_command_is_not_given_a_command():
    text = _cli("--part", "command", "--run", "triage-live-baseline")
    assert "is not an adapter run (no manifest)" in text
    judge = _cli("--part", "command", "--run", "judge-v1")
    assert "is not an adapter run (judge)" in judge


def test_an_adapter_that_reads_its_model_from_the_config_file_says_so():
    listed = _cli("--part", "list")
    live = [
        ln.split()[0] for ln in listed.splitlines()[1:-1] if ln.split()[-1] == "yes"
    ]
    names = [n for n in live if n.startswith(("reorder", "pkg"))]
    assert names, "a reorder or pkg run should be recorded"
    assert "config/models.yaml" in _cli("--part", "command", "--run", names[0])


def test_endings_counts_how_the_runs_of_a_recorded_run_ended():
    text = _cli("--part", "endings", "--run", "triage-incident-lite")
    rows = {ln[:42].strip(): int(ln[42:]) for ln in text.splitlines()[1:4]}
    assert rows == {"no error": 424, "HandoffLoopDetected": 56, "all": 480}
    assert "first error" not in text


def test_endings_warns_when_the_errors_look_like_a_key_or_a_quota():
    traces = [
        Trace(
            case_id=f"T-{i}", adapter="x", error="APIStatusError: HTTP 402 no credits"
        )
        for i in range(4)
    ] + [Trace(case_id="T-9", adapter="x")]
    text = reproduce.render_endings(traces)
    assert "4 of 5 errors mention a status code" in text
    assert "Delete the run, fix that, and run again." in text
    assert all(len(line) <= 78 for line in text.splitlines())


def test_endings_does_not_warn_about_the_products_own_errors():
    traces = [Trace(case_id="T-1", adapter="x", error="ToolLoopDidNotConverge: 12")]
    assert "first error" not in reproduce.render_endings(traces)
