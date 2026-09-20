"""Compare the shape of runs across versions of the product (chapter 10).

Usage: uv run python scripts/compare_trajectories.py MAX_CALLS MAX_STALL NAME=RUN ...

Reads recorded runs, so it needs no key.
"""

import sys
from pathlib import Path

from agent_evals import trajectory
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]


def main(max_calls: int, max_stall: int, specs: list[str]) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    print(f"{'':<20}{'no answer':>10}{'over':>6}{'stalled':>9}", end="")
    print(f"{'returned':>10}{'restarts':>10}{'calls/run':>10}")
    for spec in specs:
        name, _, run = spec.partition("=")
        traces = read_traces(ROOT / "runs" / run / "traces.jsonl")
        count = dict.fromkeys(trajectory.RULES, 0)
        for t in traces:
            for rule in trajectory.violations(
                cases[t.case_id], t, max_calls, max_stall
            ):
                count[rule] += 1
        calls = [c for t in traces for c in t.tool_calls]
        restarts = sum(c["name"] == "restart_service" for c in calls)
        print(f"{name:<20}{count['ended_without_answer']:>10}", end="")
        print(f"{count['over_call_budget']:>6}{count['stalled']:>9}", end="")
        print(f"{count['handoff_returned']:>10}{restarts:>10}", end="")
        print(f"{len(calls) / len(traces):>10.1f}")


if __name__ == "__main__":
    main(int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:])
