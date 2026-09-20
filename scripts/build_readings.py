"""Write my hand readings of the recorded answers (datasets/pkg_answers_v1.readings.jsonl).

I read every answer in runs/pkg-answers-1 and runs/pkg-answers-2 against the three
summaries the model was given, on 2026-09-20. One reader, the author, with no second
opinion. Each answer gets one reading:

- refusal: the answer says the context does not have what is needed, and cites nothing.
- supported: every claim in the answer is stated in the retrieved summaries, or is a
  plain paraphrase of them or of the question.
- stretch: the answer adds a relation or a fact that the summaries do not state.
- unsupported: the answer states something the summaries contradict or do not touch.

Refusals are found by phrase match and cross-checked: every answer that cites nothing
is a refusal and no answer that cites something is one. Every other answer is
`supported` unless it is listed in EXCEPTIONS, which holds the ones I read as `stretch`
or `unsupported`.
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
        for k in ("refusal", "supported", "stretch")
    }
    print(f"wrote {len(rows)} readings: {counts}")


if __name__ == "__main__":
    main()
