"""Where should a stall limit go? (chapter 10)

For each limit K, count the runs of the unchanged product that a guard with that
limit would have stopped: those that ended in an error, and those that went on to
answer. Reads recorded runs, so it needs no key.

Usage: uv run python scripts/budget_analysis.py RUN [RUN ...]
"""

import sys
from pathlib import Path

from agent_evals import trajectory
from agent_evals.runner import read_traces

ROOT = Path(__file__).resolve().parents[1]


def main(runs: list[str]) -> None:
    traces = [
        t for run in runs for t in read_traces(ROOT / "runs" / run / "traces.jsonl")
    ]
    failed = [t for t in traces if t.error is not None]
    answered = [t for t in traces if t.error is None]
    print(f"{len(traces)} runs: {len(failed)} ended in an error, ", end="")
    print(f"{len(answered)} ended in an answer")
    print(f"{'limit':<8}{'errors stopped':>16}{'answers interrupted':>21}")
    for limit in (2, 3, 4, 5):
        stopped = sum(trajectory.empty_rounds_in_a_row(t) >= limit for t in failed)
        cut = sum(trajectory.empty_rounds_in_a_row(t) >= limit for t in answered)
        print(f"{limit:<8}{stopped:>16}{cut:>21}")


if __name__ == "__main__":
    main(sys.argv[1:])
