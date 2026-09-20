"""Write my decisions on the answers where judges and I disagree (datasets/judge_adjudication_v1.jsonl), chapter 17.

The queue is every real answer where the majority of five judge runs, over both passes,
disagrees with my reading: I read supported and at least 4 of 10 verdicts said
unsupported, or I read borderline or stretch and at most 5 of 10 said unsupported. There
are seven, from four questions. I read each one again, with the judges' claim lists next
to it, and wrote down what I decided and why.

I am one person, and I adjudicated my own labels after seeing the judges. That is the
weakest form of adjudication there is. The original readings file is not changed. The
file here is applied on top of it, and every result in chapter 17 says which one it uses.
"""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "judge_adjudication_v1.jsonl"

ROWS = [
    {
        "item_id": "pkg-answers-1:PQ-013",
        "was": "supported",
        "decision": "borderline",
        "judges_flagged": "6 of 10",
        "reason": (
            "SQLAlchemy is offered as a way to connect to Postgres. Its summary, "
            "'Database Abstraction Library', does not mention Postgres. The judges are right."
        ),
    },
    {
        "item_id": "pkg-answers-2:PQ-013",
        "was": "supported",
        "decision": "borderline",
        "judges_flagged": "5 of 10",
        "reason": "The same claim about SQLAlchemy and Postgres, in the second run's words.",
    },
    {
        "item_id": "pkg-answers-1:PQ-042",
        "was": "stretch",
        "decision": "stretch",
        "judges_flagged": "4 of 10",
        "reason": (
            "'Similar to Flask' relates aiohttp to Flask, which no summary does. Most "
            "judges pass it. I keep my reading and mark it as one for a second person."
        ),
    },
    {
        "item_id": "pkg-answers-1:PQ-053",
        "was": "borderline",
        "decision": "borderline",
        "judges_flagged": "0 of 10",
        "reason": (
            "The purpose, keeping secrets out of code, comes from the question. No judge "
            "flags it. It is the kind of case I defined borderline for, so it stays."
        ),
    },
    {
        "item_id": "pkg-answers-2:PQ-053",
        "was": "borderline",
        "decision": "borderline",
        "judges_flagged": "0 of 10",
        "reason": "Same as the first run.",
    },
    {
        "item_id": "pkg-answers-1:PQ-059",
        "was": "borderline",
        "decision": "borderline",
        "judges_flagged": "0 of 10",
        "reason": (
            "The summary says machine learning and does not name regression. No judge "
            "flags it. It stays borderline."
        ),
    },
    {
        "item_id": "pkg-answers-2:PQ-059",
        "was": "borderline",
        "decision": "borderline",
        "judges_flagged": "0 of 10",
        "reason": "Same as the first run.",
    },
]


def main() -> None:
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in ROWS),
        encoding="utf8",
        newline="\n",
    )
    changed = sum(r["was"] != r["decision"] for r in ROWS)
    print(f"wrote {len(ROWS)} decisions, {changed} changed")


if __name__ == "__main__":
    main()
