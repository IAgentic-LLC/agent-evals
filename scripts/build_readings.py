"""Write my hand readings of the recorded answers (datasets/pkg_answers_v1.readings.jsonl).

I read every answer in runs/pkg-answers-1 and runs/pkg-answers-2 against the three
summaries the model was given, on 2026-09-20. One reader, the author, with no second
opinion. Each answer gets one reading:

- refusal: the answer says the context does not have what is needed, and cites nothing.
- supported: every claim about a package is stated in its retrieved summary, or is a
  plain rewording of it. The answer may repeat the question's own words as the purpose.
- borderline: the answer applies the question's own framing to a package in a way the
  summary does not directly state, such as a comparison, a purpose or a capability
  that the summary leaves to the reader. A second reader might say supported or stretch.
- stretch: the answer states a relation or a fact about a package that its summary does
  not state, and that I could not read as the question's own framing.
- unsupported: the answer states something the summaries contradict or do not touch.

Refusals are found by phrase match and cross-checked: every answer that cites nothing
is a refusal and no answer that cites something is one. Every other answer is
`supported` unless it is listed in EXCEPTIONS, which holds the ones I read as
`borderline`, `stretch` or `unsupported`. I read every answer, and I wrote down only the
exceptions, so the default label was never written by hand.
"""

import json
import re
from pathlib import Path

from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
RUNS = ("pkg-answers-1", "pkg-answers-2")
OUT = ROOT / "datasets" / "pkg_answers_v1.readings.jsonl"
REFUSAL = re.compile(
    r"does not contain|do not contain|not enough information", re.IGNORECASE
)

EXCEPTIONS = {
    ("pkg-answers-1", "PQ-042"): (
        "stretch",
        "Says aiohttp is 'similar to Flask'. The summaries never relate the two.",
    ),
    ("pkg-answers-1", "PQ-043"): (
        "borderline",
        (
            "Says polars 'can serve as a faster replacement for pandas'. Neither summary "
            "states a replacement or a comparison."
        ),
    ),
    ("pkg-answers-2", "PQ-043"): (
        "borderline",
        "Says polars is 'a faster option compared to pandas'. No summary compares them.",
    ),
    ("pkg-answers-1", "PQ-053"): (
        "borderline",
        (
            "Says python-dotenv keeps passwords and keys out of source code. The summary "
            "says only that it reads a .env file."
        ),
    ),
    ("pkg-answers-2", "PQ-053"): (
        "borderline",
        "Same as the first run: the purpose comes from the question, not the summary.",
    ),
    ("pkg-answers-1", "PQ-059"): (
        "borderline",
        (
            "Says scikit-learn can predict a number from many columns. The summary says "
            "machine learning and data mining and does not name regression."
        ),
    ),
    ("pkg-answers-2", "PQ-059"): (
        "borderline",
        "Same as the first run.",
    ),
}


def main() -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets" / "pkg_answers_v1.jsonl")
    }
    rows = []
    for run in RUNS:
        for trace in read_traces(ROOT / "runs" / run / "traces.jsonl"):
            refused = bool(REFUSAL.search(trace.answer))
            assert refused == (not trace.cited), (run, trace.case_id)
            if refused:
                reading, note = "refusal", ""
            else:
                reading, note = EXCEPTIONS.get((run, trace.case_id), ("supported", ""))
            assert trace.case_id in cases
            rows.append(
                {"run": run, "case_id": trace.case_id, "reading": reading, "note": note}
            )
    OUT.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf8",
        newline="\n",
    )
    counts = {
        k: sum(r["reading"] == k for r in rows)
        for k in ("refusal", "supported", "borderline", "stretch")
    }
    print(f"wrote {len(rows)} readings: {counts}")


if __name__ == "__main__":
    main()
