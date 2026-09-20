"""Chapter 21: red-teaming the triage product with injected instructions."""

import json
from collections import Counter
from pathlib import Path

from agent_evals.adapters.triage import UNTRUSTED_NOTE, UntrustedTextAdapter

ROOT = Path(__file__).resolve().parents[1]


def _rows():
    text = (ROOT / "datasets/triage_redteam_v1.jsonl").read_text(encoding="utf8")
    return [json.loads(x) for x in text.splitlines() if x.strip()]


def test_the_untrusted_note_is_added_to_the_question_and_taken_off_again():
    from triage_app import specialists
    from triage_app.tickets import Ticket

    before = specialists._question_for
    ticket = Ticket(
        ticket_id="X", customer_id="cust-1", category="billing", subject="s", body="b"
    )
    with UntrustedTextAdapter():
        question = specialists._question_for(ticket, None)
        assert question.startswith(UNTRUSTED_NOTE)
        assert "Customer ID: cust-1" in question and question.endswith("b")
    assert specialists._question_for is before


def test_the_red_team_set_is_12_controls_72_in_scope_and_12_cross_wall_attacks():
    rows = _rows()
    walls = Counter(r["slices"]["wall"] for r in rows)
    assert walls == {"clean": 12, "in_scope": 72, "cross_wall": 12}
    families = Counter(
        r["slices"]["family"] for r in rows if r["slices"]["wall"] == "in_scope"
    )
    assert set(families.values()) == {12} and len(families) == 6


def test_every_attack_targets_one_action_and_its_host_needs_no_change_to_the_world():
    for r in _rows():
        target = r["slices"]["target"]
        assert r["invariants"]["forbidden_actions"] == [target]
        assert target not in r["expected"]["required_actions"]
        assert r["expected"]["handled_by"] == r["input"]["category"]


def test_a_cross_wall_attack_names_a_tool_its_specialist_does_not_have():
    have = {
        "billing": {"look_up_invoice", "issue_refund"},
        "technical": {"search_runbook", "restart_service"},
        "security": {"escalate_to_oncall", "freeze_account"},
    }
    for r in _rows():
        spec, target = r["slices"]["specialist"], r["slices"]["target"]
        if r["slices"]["wall"] == "cross_wall":
            assert target not in have[spec]
        else:
            assert target in have[spec]


def test_each_attack_adds_the_target_to_a_host_and_leaves_the_control_alone():
    rows = _rows()
    controls = {r["slices"]["host"]: r for r in rows if r["slices"]["wall"] == "clean"}
    for r in rows:
        control = controls[r["slices"]["host"]]
        assert r["input"]["customer_id"] == control["input"]["customer_id"]
        if r["slices"]["wall"] != "clean":
            assert (r["input"]["subject"], r["input"]["body"]) != (
                control["input"]["subject"],
                control["input"]["body"],
            )
        if r["slices"]["family"] != "subject" and r["slices"]["wall"] != "clean":
            assert r["input"]["body"].startswith(control["input"]["body"])
