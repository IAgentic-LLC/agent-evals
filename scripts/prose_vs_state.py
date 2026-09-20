"""Compare two prose graders with the recorded state, for four actions (chapter 6).

For every recorded answer, the state says whether the action happened. A prose grader
should agree. Runs that ended in an error or have no answer are left out, since there
is no prose to read.

Usage: uv run python scripts/prose_vs_state.py
"""

from pathlib import Path

from agent_evals.action_claims import (
    ACTIONS,
    claims_action_done,
    compare,
    did_action,
    mentions_action,
)
from agent_evals.runner import read_traces

RUNS = Path(__file__).resolve().parents[1] / "runs"


def main() -> None:
    traces = [t for p in sorted(RUNS.glob("*/traces.jsonl")) for t in read_traces(p)]
    answered = sum(1 for t in traces if not t.error and t.answer.strip())
    print(f"{answered} recorded answers")
    print(
        f"{'action':<20}{'grader':<10}{'both':>6}{'prose only':>12}{'state only':>12}"
    )
    for action in ACTIONS:
        state = did_action(action)
        for name, prose in (
            ("mentions", mentions_action(action)),
            ("claims", claims_action_done(action)),
        ):
            c = compare(prose, state, traces)
            print(
                f"{action:<20}{name:<10}{c['both']:>6}"
                f"{c['prose_only']:>12}{c['state_only']:>12}"
            )


if __name__ == "__main__":
    main()
