"""Chapter 5: graders are software, so they are tested like software.

No API key needed: the recorded runs and my hand labels.
"""

from collections import Counter
from pathlib import Path

import pytest

from agent_evals.answer_graders import (
    ASKS_FOR_KNOWN_INFO,
    acted_on_the_right_customer,
    asks_v1,
    asks_v2,
    asks_v3,
    asks_v4,
    asks_v5,
)
from agent_evals.cli import main
from agent_evals.grader_check import check, load_cases_by_id, load_labels
from agent_evals.runner import read_traces
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "datasets" / "graders" / "asks_for_known_info.labels.jsonl"
CASES = load_cases_by_id(
    ROOT / "datasets/triage_book3_six.jsonl", ROOT / "datasets/triage_heldout_v1.jsonl"
)


def _case(customer="cust-42"):
    return EvalCase(case_id="c", input={"customer_id": customer}, expected={})


def _trace(answer="", actions=()):
    return Trace(case_id="c", adapter="t", answer=answer, actions_taken=list(actions))


def test_the_hand_labels_have_the_splits_the_chapter_describes():
    labels = load_labels(LABELS)
    by_split = Counter(row["split"] for row in labels)
    assert by_split == {"dev": 117, "test": 59}
    positives = Counter(row["split"] for row in labels if row["label"])
    assert positives == {"dev": 36, "test": 16}
    assert sum(row["contested"] for row in labels) == 5


# (true positives, false positives, false negatives, true negatives)
DEV = {
    "v1": (29, 6, 7, 75),
    "v2": (32, 3, 4, 78),
    "v3": (32, 0, 4, 81),
    "v4": (34, 0, 2, 81),
    "v5": (35, 2, 1, 79),
}
TEST = {
    "v1": (14, 6, 2, 37),
    "v2": (13, 2, 3, 41),
    "v3": (13, 1, 3, 42),
    "v4": (15, 1, 1, 42),
    "v5": (15, 1, 1, 42),
}


@pytest.mark.parametrize("split,expected", [("dev", DEV), ("test", TEST)])
def test_each_grader_version_scores_what_the_chapter_reports(split, expected):
    part = [row for row in load_labels(LABELS) if row["split"] == split]
    for version, grader in ASKS_FOR_KNOWN_INFO.items():
        r = check(grader, part, ROOT / "runs", CASES)
        assert (r.tp, r.fp, r.fn, r.tn) == expected[version], version


def test_a_plain_request_for_the_customer_id_is_flagged_and_using_it_is_not():
    ask = _trace(
        "Could you please provide your Customer ID so I can look up your invoice?"
    )
    used = _trace("I checked your account (`cust-42`) and found your invoice.")
    assert asks_v1(_case(), ask) and asks_v2(_case(), ask)
    assert asks_v1(_case(), used) is False and asks_v2(_case(), used) is False


def test_v1_wrongly_flags_any_mention_of_the_phrase():
    mention = _trace("Your Customer ID is not needed, I already have it.")
    assert asks_v1(_case(), mention)
    assert asks_v2(_case(), mention) is False


def test_v3_stays_quiet_when_the_specialist_already_used_the_ticket_id():
    text = "Please provide any receipt number or the email address on the payment."
    looked_up = _trace(text, [{"action": "look_up_invoice", "customer_id": "cust-42"}])
    assert asks_v2(_case(), looked_up) and not asks_v3(_case(), looked_up)


def test_v4_reads_spanish_and_french_and_v3_does_not():
    es = _trace("Para ayudarte, podrias proporcionarme tu ID de cliente?")
    fr = _trace("Pourriez-vous me fournir votre identifiant client ?")
    for answer in (es, fr):
        assert asks_v4(_case(), answer) and not asks_v3(_case(), answer)


def test_the_known_false_positive_asking_for_other_peoples_ids_is_flagged():
    other = _trace("Please provide any error logs, user IDs, or timestamps.")
    assert asks_v4(_case(), other)  # a known error, kept on record


def test_v5_trades_a_missed_ask_for_a_false_alarm():
    reply = _trace("If you know your account ID, please reply with it.")
    former = _trace("For the former employee's email addresses, please share them.")
    assert asks_v5(_case(), reply) and not asks_v4(_case(), reply)
    assert asks_v5(_case(), former) and not asks_v4(_case(), former)


def test_the_customer_check_passes_on_every_recorded_run_and_fails_on_a_planted_mistake():
    traces = [
        t
        for path in (ROOT / "runs").glob("*/traces.jsonl")
        if not path.parent.name.endswith("-leaky")  # a known-bad fixture
        for t in read_traces(path)
        if t.case_id in CASES and t.actions_taken
    ]
    assert len(traces) >= 183  # later chapters add runs
    assert all(acted_on_the_right_customer(CASES[t.case_id], t) for t in traces)
    planted = _trace(actions=[{"action": "freeze_account", "customer_id": "cust-99"}])
    assert acted_on_the_right_customer(_case("cust-42"), planted) is False
    assert acted_on_the_right_customer(_case("cust-42"), _trace()) is True


def test_the_cli_reports_the_check_and_the_grade(capsys):
    main(
        [
            "grader",
            "check",
            "--grader",
            "asks_for_known_info:v4",
            "--labels",
            str(LABELS),
            "--split",
            "test",
        ]
    )
    out = capsys.readouterr().out
    assert "Precision  15/16" in out and "Recall     15/16" in out
    dataset = str(ROOT / "datasets/triage_heldout_v1.jsonl")
    for run, expected in (
        ("triage-heldout-v1-shipped", "16 of 33"),
        ("triage-heldout-v1", "0 of 30"),
    ):
        main(
            [
                "grade",
                "--run",
                str(ROOT / "runs" / run),
                "--dataset",
                dataset,
                "--grader",
                "asks_for_known_info:v4",
            ]
        )
        assert f"flagged {expected}" in capsys.readouterr().out
