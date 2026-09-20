"""What the runbook search returned, by the ticket's runbook topic (chapter 9).

Usage: uv run python scripts/search_outcomes.py NAME=RUN[,RUN...] [NAME=RUN...]

Each group is one or more recorded runs of the same tickets. The script reads their
recorded tool calls, so it needs no key.
"""

import sys
from collections import Counter
from pathlib import Path

from agent_evals.runner import load_cases, read_traces
from agent_evals.tool_calls import search_outcome

ROOT = Path(__file__).resolve().parents[1]


def main(specs: list[str]) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    widths = {"runs": 6, "searches": 10, "relevant": 10, "irrelevant": 12, "empty": 7}
    header = f"{'':<14}" + "".join(f"{k:>{w}}" for k, w in widths.items())
    print(header + f"{'errors':>8}")
    for spec in specs:
        name, _, runs = spec.partition("=")
        print(name)
        rows: dict[str, Counter] = {}
        for run in runs.split(","):
            for t in read_traces(ROOT / "runs" / run / "traces.jsonl"):
                topic = cases[t.case_id].slices["runbook_topic"]
                row = rows.setdefault(topic, Counter())
                row["runs"] += 1
                row["errors"] += t.error is not None
                for call in t.tool_calls:
                    kind = search_outcome(cases[t.case_id], call)
                    if kind:
                        row["searches"] += 1
                        row[kind] += 1
        for topic in ("crash", "login", "other", "na"):
            row = rows.get(topic, Counter())
            line = f"  {topic:<12}" + "".join(
                f"{row[k]:>{w}}" for k, w in widths.items()
            )
            print(line + f"{row['errors']:>8}")


if __name__ == "__main__":
    main(sys.argv[1:])
