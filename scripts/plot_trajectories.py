"""Plot how long runs were and how they ended, before and after the guard (figure 10.2).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_trajectories.py OUTPUT.png
"""

import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.runner import read_traces

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
ENDINGS = (
    ("answer", "ended in an answer", "#2563EB"),
    ("ToolLoopDidNotConverge", "hit the round limit", "#E03131"),
    ("HandoffLoopDetected", "handed back and forth", "#F08C00"),
)
PANELS = (
    (
        "Unchanged product, two passes",
        ("triage-heldout-v1-calls", "triage-heldout-v1-calls-2"),
    ),
    (
        "With the stall guard, two passes",
        ("triage-heldout-v1-guard", "triage-heldout-v1-guard-2"),
    ),
)


def main(out: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 4.4), dpi=200, sharex=True)
    for ax, (title, runs) in zip(axes, PANELS):
        counts: Counter = Counter()
        for run in runs:
            for t in read_traces(ROOT / "runs" / run / "traces.jsonl"):
                ending = (t.error or "answer").split(":")[0]
                counts[(len(t.tool_calls), ending)] += 1
        bottom = [0] * 7
        for key, label, color in ENDINGS:
            heights = [counts[(n, key)] for n in range(7)]
            ax.bar(
                range(7), heights, bottom=bottom, color=color, label=label, width=0.7
            )
            bottom = [b + h for b, h in zip(bottom, heights)]
        for n, total in enumerate(bottom):
            if total:
                ax.text(n, total + 1, str(total), ha="center", fontsize=9, color=INK)
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.set_ylim(0, 48)
        ax.set_xlim(0.4, 6.6)
        ax.set_xticks(range(1, 7))
        ax.tick_params(labelsize=9, colors=INK)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[1].set_xlabel("Tool calls in the run", fontsize=10, color=INK)
    axes[0].set_ylabel("Runs", fontsize=10, color=INK)
    axes[1].set_ylabel("Runs", fontsize=10, color=INK)
    axes[0].legend(fontsize=9, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
