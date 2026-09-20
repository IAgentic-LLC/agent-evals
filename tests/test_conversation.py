"""Chapter 11: a conversation that pauses for a human and resumes.

No API key needed: a scripted model, hand-made traces, and the recorded runs.
"""

import re
from pathlib import Path

import pytest
from reliable_agents_labs.inventory import check_inventory

from agent_evals import cli, conversation, mechanics
from agent_evals.adapters.reorder import ReorderScriptedAdapter
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/reorder_conversations_v1.jsonl"
CASES = {c.case_id: c for c in load_cases(DATASET)}
RUNS = ("reorder-conversations-1", "reorder-conversations-2")


async def _failed(name: str) -> set[str]:
    found = await mechanics.run_mechanics(mechanics.VARIANTS[name])
    return {check for check, why in found.items() if why}


async def test_the_real_workflow_passes_every_mechanics_check():
    assert await _failed("the real workflow") == set()


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("pauses inside the model node", {"resume_asks_the_model_nothing"}),
        ("forgets on restart", {"state_survives_a_restart"}),
        ("one thread for everyone", {"threads_do_not_mix"}),
        (
            "logs the action twice",
            {
                "approval_acts_once",
                "state_survives_a_restart",
                "a_second_approval_acts_once",
                "threads_do_not_mix",
            },
        ),
    ],
)
async def test_each_planted_fault_fails_the_checks_that_should_catch_it(
    variant, expected
):
    assert await _failed(variant) == expected


async def test_a_workflow_that_acts_before_approval_fails_seven_of_eight_checks():
    failed = await _failed("acts before approval")
    assert "pauses_before_acting" in failed and "rejection_does_not_act" in failed
    assert len(failed) == 7 and "does_not_pause_when_not_needed" not in failed


async def test_a_scripted_conversation_records_what_each_turn_did():
    case = _case("Do we need to reorder SKU-1029?", "approve")
    trace = await ReorderScriptedAdapter().run(case, 1)
    ask, decide = trace.turns
    assert (ask["sent"], ask["paused"], ask["model_calls"], ask["logged"]) == (
        "ask",
        True,
        3,
        False,
    )
    assert ask["next"] == ["await_approval"]
    assert (
        decide["sent"],
        decide["paused"],
        decide["model_calls"],
        decide["logged"],
    ) == (
        "approve",
        False,
        0,
        True,
    )
    assert [a["action"] for a in trace.actions_taken] == ["log_reorder"]


async def test_a_rejection_or_a_question_that_needs_nothing_does_not_act():
    rejected = await ReorderScriptedAdapter().run(
        _case("Do we need to reorder SKU-1029?", "reject"), 1
    )
    fine = await ReorderScriptedAdapter().run(
        _case("Do we need to reorder SKU-2040?", "approve"), 1
    )
    assert rejected.actions_taken == [] and len(rejected.turns) == 2
    assert fine.actions_taken == [] and len(fine.turns) == 1


def _case(question: str, decision: str) -> EvalCase:
    needing = [
        s
        for s in dict.fromkeys(re.findall(r"SKU-\d+", question))
        if (r := check_inventory(s)) and r.quantity < r.reorder_point
    ]
    return EvalCase(
        case_id="X",
        input={"question": question, "decision": decision},
        expected={"needs_reorder": bool(needing), "skus_needing_reorder": needing},
        slices={"kind": "low"},
    )


def test_the_dataset_is_labeled_from_the_inventory():
    assert len(CASES) == 26 and len({c.case_id for c in CASES.values()}) == 26
    assert all(c.canary and c.split == "test" for c in CASES.values())
    contested = {
        c.case_id for c in CASES.values() if c.slices["label_confidence"] == "contested"
    }
    assert contested == {"RC-020", "RC-021", "RC-025"}
    for c in CASES.values():
        expected = _case(c.input["question"], c.input["decision"]).expected
        assert c.expected == expected
    kinds = {}
    for c in CASES.values():
        kinds[c.slices["kind"]] = kinds.get(c.slices["kind"], 0) + 1
    assert kinds == {
        "low": 7,
        "fine": 5,
        "unknown": 2,
        "multi": 5,
        "vague": 2,
        "injection": 3,
        "info": 2,
    }


def _trace(
    case_id, *, paused=True, logged_early=False, decision_logged=True, resume_calls=0
):
    turns = [
        {
            "turn": 1,
            "sent": "ask",
            "paused": paused,
            "next": [],
            "model_calls": 3,
            "reorder": paused,
            "logged": logged_early,
        },
    ]
    if paused:
        turns.append(
            {
                "turn": 2,
                "sent": "decide",
                "paused": False,
                "next": [],
                "model_calls": resume_calls,
                "reorder": True,
                "logged": decision_logged,
            }
        )
    q = CASES[case_id].input["question"]
    acts = (
        [{"action": "log_reorder", "question": q}]
        if (decision_logged or logged_early)
        else []
    )
    return Trace(case_id=case_id, adapter="t", actions_taken=acts, turns=turns)


def test_acting_before_any_decision_is_flagged():
    case = CASES["RC-001"]
    bad = _trace("RC-001", logged_early=True)
    assert conversation.grade(case, bad)["acted_before_approval"] is True
    assert conversation.violated_invariants(case, bad) == ["acted_before_approval"]


def test_acting_after_a_rejection_is_flagged():
    case = CASES["RC-002"]  # the decision in this case is to reject
    assert case.input["decision"] == "reject"
    bad = _trace("RC-002", decision_logged=True)
    assert conversation.grade(case, bad)["acted_after_rejection"] is True
    ok = _trace("RC-002", decision_logged=False)
    assert conversation.violated_invariants(case, ok) == []


def test_a_resume_that_calls_the_model_is_flagged_and_a_missed_action_too():
    case = CASES["RC-001"]
    assert conversation.grade(case, _trace("RC-001", resume_calls=3))[
        "asked_model_on_resume"
    ]
    missed = _trace("RC-001", decision_logged=False)
    assert conversation.grade(case, missed)["missed_after_approval"] is True


def test_the_order_the_product_places_follows_the_first_sku_named():
    # RC-015 names SKU-2040, which is well stocked, before SKU-1029, which is not.
    case, trace = CASES["RC-015"], _trace("RC-015")
    assert conversation.order_the_product_would_place(trace) == "SKU-2040"
    assert conversation.grade(case, trace)["wrong_sku_ordered"] is True
    # RC-016 names SKU-1029 first, so the same rule happens to be right.
    assert (
        conversation.grade(CASES["RC-016"], _trace("RC-016"))["wrong_sku_ordered"]
        is False
    )


def _traces(run):
    return read_traces(ROOT / "runs" / run / "traces.jsonl")


def _totals():
    rows = [
        conversation.grade(CASES[t.case_id], t) for run in RUNS for t in _traces(run)
    ]

    def count(measure):
        applies = [r[measure] for r in rows if r[measure] is not None]
        return sum(applies), len(applies)

    return {m: count(m) for m in conversation.MEASURES}


def test_the_recorded_conversations_hold_the_invariants_and_pick_the_wrong_sku_sometimes():
    assert _totals() == {
        "paused_right": (50, 52),
        "acted_before_approval": (0, 52),
        "acted_after_rejection": (0, 11),
        "missed_after_approval": (0, 15),
        "asked_model_on_resume": (0, 26),
        "wrong_sku_ordered": (4, 15),
    }


def test_the_wrong_orders_are_the_two_multi_sku_questions_in_both_passes():
    wrong = [
        (run[-1], t.case_id)
        for run in RUNS
        for t in _traces(run)
        if conversation.grade(CASES[t.case_id], t)["wrong_sku_ordered"]
    ]
    assert sorted(wrong) == [
        ("1", "RC-015"),
        ("1", "RC-017"),
        ("2", "RC-015"),
        ("2", "RC-017"),
    ]


def test_the_two_missed_pauses_are_answers_that_report_stock_without_recommending():
    missed = [
        (run[-1], t.case_id)
        for run in RUNS
        for t in _traces(run)
        if conversation.grade(CASES[t.case_id], t)["paused_right"] is False
    ]
    assert sorted(missed) == [("2", "RC-004"), ("2", "RC-023")]


def test_every_recorded_conversation_used_three_model_calls_to_ask_and_none_to_resume():
    for run in RUNS:
        for t in _traces(run):
            assert t.error is None
            assert t.turns[0]["model_calls"] >= 2
            if len(t.turns) > 1:
                assert t.turns[1]["model_calls"] == 0


def test_the_conversation_commands_exit_0_on_the_recorded_runs(capsys):
    assert cli.main(["conversation", "check"]) == 0
    args = ["conversation", "grade", "--dataset", str(DATASET), "--run"]
    assert cli.main(args + [str(ROOT / "runs" / r) for r in RUNS]) == 0
    assert "50 of 52" in capsys.readouterr().out


def test_a_reorder_manifest_names_the_model_and_the_reorder_app():
    from datetime import UTC, datetime

    from agent_evals.manifest import build_manifest

    manifest = build_manifest(
        harness=None,
        adapter="reorder-live",
        dataset=DATASET,
        cases=26,
        trials=1,
        concurrency=1,
        started_at=datetime.now(UTC),
        wall_seconds=1.0,
    )
    assert manifest["model"]["model_id"] == "gemini-3.6-flash"
    assert "reorder-app" in manifest["packages"]
