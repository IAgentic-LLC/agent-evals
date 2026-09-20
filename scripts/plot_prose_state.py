"""Plot how often a prose grader and the recorded state agree (chapter 6, figure 6.2).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_prose_state.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from agent_evals.action_claims import (
    ACTIONS,
    claims_action_done,
    compare,
    did_action,
    mentions_action,
)
from agent_evals.runner import read_traces

RUNS = Path(__file__).resolve().parents[1] / "runs"
BLUE, RED, ORANGE, INK = "#2563EB", "#E03131", "#E8590C", "#1F2937"
LABELS = {
    "issue_refund": "refund",
    "freeze_account": "freeze",
    "escalate_to_oncall": "escalate",
    "restart_service": "restart",
}


def main(out: str) -> None:
    traces = [t for p in sorted(RUNS.glob("*/traces.jsonl")) for t in read_traces(p)]
    rows = []
    for action in ACTIONS:
        state = did_action(action)
        for kind, prose in (
            ("mentions", mentions_action(action)),
            ("claims", claims_action_done(action)),
        ):
            rows.append((f"{LABELS[action]}, {kind}", compare(prose, state, traces)))
    fig, ax = plt.subplots(figsize=(5.5, 3.6), dpi=200)
    for i, (label, c) in enumerate(reversed(rows)):
        left = 0
        for key, color in (("both", BLUE), ("prose_only", RED), ("state_only", ORANGE)):
            n = c[key]
            if n:
                ax.barh(i, n, left=left, color=color, height=0.62)
                if n >= 3:
                    ax.text(
                        left + n / 2,
                        i,
                        str(n),
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="white",
                    )
            left += n
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10, color=INK)
    ax.set_xlabel("Answers", fontsize=10, color=INK)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [
        Patch(color=BLUE, label="prose and state agree"),
        Patch(color=RED, label="prose says it, state does not"),
        Patch(color=ORANGE, label="state has it, prose does not"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=9, loc="lower right")
    ax.text(0.6, 0, "no answer differs", va="center", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
