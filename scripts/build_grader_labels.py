"""Write my hand labels for the "asks for known information" grader (chapter 5).

Rule. An answer is positive if it asks the customer to identify their account
(customer ID, account ID, username or account email) as a next step, in any
language, when the ticket record already carries a customer ID and the specialist
did not try that ID first. An answer that only mentions the ID, that asks for
other information (a device model, a receipt number after a failed lookup), or
that asks about someone else's account is negative.

I went through every answer in the recorded runs below, reading each sentence
that mentions an account, an email or an ID, or asks a question. Every answer
not listed as positive is negative. `contested` marks a label I am not sure of.

One labeler, so these labels carry one person's judgment.
"""

import json
from pathlib import Path

from agent_evals.runner import read_traces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "graders" / "asks_for_known_info.labels.jsonl"

RUNS = [
    "triage-heldout-v1",
    "triage-heldout-v1-shipped",
    "triage-live-shipped",
    "triage-live-customer-id",
    "triage-dev-six-ch04",
    "triage-live-3x-sequential",
    "triage-live-3x-concurrent",
]

# The test split: two runs recorded after the grader versions were frozen, labeled
# by reading the answers before any grader was run on them.
TEST_RUNS = ["triage-heldout-v1-shipped-2", "triage-heldout-v1-2"]
TEST_POSITIVE = {
    ("triage-heldout-v1-shipped-2", case): None
    for case in (
        "HO-001",
        "HO-002",
        "HO-003",
        "HO-004",
        "HO-005",
        "HO-006",
        "HO-007",
        "HO-008",
        "HO-009",
        "HO-010",
        "HO-023",
        "HO-024",
        "HO-026",
        "HO-032",
        "HO-036",
        "HO-040",
    )
}
TEST_CONTESTED = {
    (
        "triage-heldout-v1-shipped-2",
        "HO-007",
        1,
    ): "asks only if the customer wants a direct check",
    (
        "triage-heldout-v1-2",
        "HO-008",
        1,
    ): "asks the customer to double-check the ID after a lookup found nothing",
}

# (run, case_id) -> the trials that are positive. None means every trial.
POSITIVE = {
    ("triage-heldout-v1-shipped", "HO-001"): None,
    ("triage-heldout-v1-shipped", "HO-002"): None,
    ("triage-heldout-v1-shipped", "HO-003"): None,
    ("triage-heldout-v1-shipped", "HO-004"): None,
    ("triage-heldout-v1-shipped", "HO-005"): None,
    ("triage-heldout-v1-shipped", "HO-006"): None,
    ("triage-heldout-v1-shipped", "HO-007"): None,
    ("triage-heldout-v1-shipped", "HO-008"): None,
    ("triage-heldout-v1-shipped", "HO-009"): None,  # Spanish
    ("triage-heldout-v1-shipped", "HO-010"): None,  # French
    ("triage-heldout-v1-shipped", "HO-023"): None,
    ("triage-heldout-v1-shipped", "HO-024"): None,
    ("triage-heldout-v1-shipped", "HO-026"): None,
    ("triage-heldout-v1-shipped", "HO-032"): None,
    ("triage-heldout-v1-shipped", "HO-036"): None,
    ("triage-heldout-v1-shipped", "HO-040"): None,
    ("triage-heldout-v1-shipped", "HO-041"): None,
    ("triage-live-shipped", "TCK-1001"): None,
    ("triage-live-shipped", "TCK-1006"): None,
    ("triage-live-3x-sequential", "TCK-1001"): None,
    ("triage-live-3x-sequential", "TCK-1004"): None,
    ("triage-live-3x-sequential", "TCK-1006"): None,
    ("triage-live-3x-concurrent", "TCK-1001"): None,
    ("triage-live-3x-concurrent", "TCK-1004"): {1, 3},
    ("triage-live-3x-concurrent", "TCK-1006"): None,
}

# Labels I am not sure of.
CONTESTED = {
    (
        "triage-heldout-v1",
        "HO-008",
        1,
    ): "asks the customer to double-check the ID after a lookup found nothing",
    (
        "triage-heldout-v1-shipped",
        "HO-007",
        1,
    ): "asks only if the charge posts, a conditional request",
    (
        "triage-live-3x-concurrent",
        "TCK-1004",
        1,
    ): "asks only if the customer is able to provide it",
}


def _rows(runs, positives, contested, split):
    rows = []
    for run in runs:
        for t in read_traces(ROOT / "runs" / run / "traces.jsonl"):
            if t.error or not t.answer.strip():
                continue
            key = (run, t.case_id)
            trials = positives.get(key, set())
            positive = key in positives and (trials is None or t.trial in trials)
            note = contested.get((run, t.case_id, t.trial), "")
            rows.append(
                {
                    "run": run,
                    "case_id": t.case_id,
                    "trial": t.trial,
                    "split": split,
                    "label": positive,
                    "contested": bool(note),
                    "note": note,
                }
            )
    return rows


def main() -> None:
    rows = _rows(RUNS, POSITIVE, CONTESTED, "dev")
    rows += _rows(TEST_RUNS, TEST_POSITIVE, TEST_CONTESTED, "test")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(
        "\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf8", newline="\n"
    )
    for split in ("dev", "test"):
        part = [r for r in rows if r["split"] == split]
        yes = sum(r["label"] for r in part)
        print(f"{split}: {len(part)} labels, {yes} positive")
    print(f"wrote {len(rows)} labels to {OUT}")


if __name__ == "__main__":
    main()
