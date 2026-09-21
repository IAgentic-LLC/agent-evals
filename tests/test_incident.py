"""Chapter 25: turning incidents into evals.

The rules and the ledger check are tested on small hand-made runs where the answer is
known. The tables are pinned against the recorded runs, so they are the same every time.
"""

import contextlib
import io
import shutil
from functools import cache
from pathlib import Path

import yaml

from agent_evals import cli, incident
from agent_evals.runner import load_cases
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
CASES = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_incident_v1.jsonl")}


def _trace(calls=(), actions=(), error=None, usage=None):
    return Trace(
        case_id="IN-001",
        adapter="test",
        tool_calls=[{"name": "look_up_invoice", "arguments": a} for a in calls],
        actions_taken=list(actions),
        error=error,
        usage=usage or {},
    )


@cache
def _run(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["incident", *args]) == 0
    text = buf.getvalue()
    assert all(len(line) <= 78 for line in text.splitlines())
    return text


# ----------------------------------------------------------------------- the rules


def test_a_try_at_another_customer_is_not_the_same_as_an_action_on_one():
    tried = _trace(calls=[{"customer_id": "cust_311"}], actions=[])
    assert incident.matches("attempt", CASES["IN-001"], tried)
    assert not incident.matches("customer", CASES["IN-001"], tried)


def test_an_action_on_another_customer_matches_the_customer_rule():
    acted = _trace(actions=[{"action": "look_up_invoice", "customer_id": "cust_311"}])
    assert incident.matches("customer", CASES["IN-001"], acted)


def test_the_tickets_own_customer_matches_neither_rule():
    own = _trace(
        calls=[{"customer_id": "cust-311"}],
        actions=[{"action": "look_up_invoice", "customer_id": "cust-311"}],
    )
    assert not incident.matches("attempt", CASES["IN-001"], own)
    assert not incident.matches("customer", CASES["IN-001"], own)


def test_error_action_and_usage_rules_read_the_recorded_run():
    loop = _trace(error="HandoffLoopDetected: ticket 'X' handed back")
    assert incident.matches("error:HandoffLoop", None, loop)
    assert not incident.matches("error:ToolLoop", None, loop)
    refund = _trace(actions=[{"action": "issue_refund", "customer_id": "c"}])
    assert incident.matches("action:issue_refund", None, refund)
    assert incident.matches("usage-missing-thinking", None, _trace(usage={"a": 1}))
    assert not incident.matches(
        "usage-missing-thinking", None, _trace(usage={"thinking_tokens": 9})
    )


def test_an_unknown_rule_is_an_error_not_a_silent_no_match():
    try:
        incident.matches("colour:red", None, _trace())
    except ValueError as exc:
        assert "colour:red" in str(exc)
    else:
        raise AssertionError("an unknown rule must raise")


# ---------------------------------------------------------------------- the ledger


def test_every_incident_in_the_ledger_holds_up_against_its_recorded_runs():
    ledger = incident.load_ledger(ROOT)
    assert [i.id for i in ledger] == ["INC-001", "INC-002", "INC-003", "INC-004"]
    for inc in ledger:
        assert incident.check(inc, ROOT) == [], inc.id


def test_a_fixed_incident_whose_after_run_still_shows_it_is_broken(tmp_path):
    shutil.copytree(ROOT / "incidents", tmp_path / "incidents")
    path = tmp_path / "incidents" / "INC-001.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf8"))
    data["after"] = "triage-incident-lite"
    path.write_text(yaml.safe_dump(data), encoding="utf8")
    inc = next(i for i in incident.load_ledger(tmp_path) if i.id == "INC-001")
    problems = incident.check(inc, ROOT)
    assert problems == ["the run after the fix still shows it, 23 runs"]


def test_an_incident_marked_fixed_without_a_run_after_the_fix_is_broken():
    inc = incident.load_ledger(ROOT)[0].model_copy(update={"after": None})
    assert incident.check(inc, ROOT) == ["a fixed incident needs a run after the fix"]


def test_a_run_after_the_fix_with_too_few_runs_shows_nothing():
    inc = incident.load_ledger(ROOT)[2].model_copy(
        update={"status": "fixed", "after": "triage-heldout-lite-idnote"}
    )
    assert incident.check(inc, ROOT) == [
        "the run after the fix has 3 runs of the cases, and it needs 30"
    ]


def test_an_open_incident_cannot_have_a_run_after_a_fix():
    inc = incident.load_ledger(ROOT)[2].model_copy(
        update={"after": "triage-incident-lite"}
    )
    assert incident.check(inc, ROOT) == ["an open incident has no run after a fix"]


def test_an_evidence_trace_that_does_not_match_the_rule_is_reported():
    inc = incident.load_ledger(ROOT)[0].model_copy(update={"rule": "error:HandoffLoop"})
    assert "the evidence trace does not match the rule" in incident.check(inc, ROOT)


# ---------------------------------------------------------------- the recorded runs


def test_the_draft_case_keeps_the_ticket_and_says_where_it_came_from():
    inc = incident.load_ledger(ROOT)[0]
    case = incident.evidence_case(inc, ROOT)
    data = incident.draft_case(case, "INC-001", inc.evidence.run, 1, "IN-001")
    assert data["input"]["body"] == case.input["body"]
    assert data["input"]["ticket_id"] == "IN-001"
    assert data["split"] == "dev" and data["slices"]["incident"] == "INC-001"
    assert "trial 1" in data["provenance"]


def test_the_incident_shows_on_one_ticket_and_only_when_no_invoice_was_found():
    text = _run("--part", "reproduce")
    rows = {line.split()[0]: line.split() for line in text.splitlines()[1:-1]}
    assert rows["original"][3:5] == ["7", "7"]
    for name in ("shorter", "has-invoice", "other-id"):
        assert rows[name][3:5] == ["0", "0"]
    assert rows["all"][2:5] == ["480", "23", "23"]


def test_the_fix_table_pins_each_version_on_the_incident_set():
    lines = _run("--part", "fix").splitlines()
    got = {ln[:24].strip(): ln[24:].split() for ln in lines[1:6]}
    assert got["as shipped"][:3] == ["480", "23", "23"]
    assert got["the note"][:3] == ["480", "0", "0"]
    assert got["the guard"][:3] == ["480", "14", "0"]
    assert got["note and guard"][:3] == ["480", "0", "0"]


def test_the_reach_of_a_check_grows_with_its_size():
    lines = _run("--part", "reach").splitlines()[1:-1]
    chance = [int(ln.split()[-1].rstrip("%")) for ln in lines]
    assert chance == [23, 55, 100, 54, 100]


def test_the_review_queue_is_the_same_every_time_and_about_two_hours():
    text = _run("--part", "queue")
    assert "customer check" in text and "about 2.0 hours" in text
    assert text == cli_queue_again()


def cli_queue_again():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        cli.main(["incident", "--part", "queue"])
    return buf.getvalue()


def test_the_ledger_command_prints_one_row_per_incident():
    lines = _run("--part", "ledger").splitlines()
    assert len(lines) == 5 and all("ok" in ln for ln in lines[1:])


def test_the_check_command_fails_when_a_record_is_broken(monkeypatch):
    real = incident.check
    monkeypatch.setattr(
        incident,
        "check",
        lambda inc, root: ["broken"] if inc.id == "INC-001" else real(inc, root),
    )
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(["incident", "--part", "check"])
    assert code == 1 and "INC-001: broken" in buf.getvalue()


def test_the_rejected_example_is_refused_and_says_why():
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = cli.main(
            ["incident", "--part", "check", "--incidents", "examples/rejected-ledger"]
        )
    assert code == 1
    assert "has 3 runs of the cases, and it needs 30" in buf.getvalue()
