"""Plot the two judge prompts side by side on the unseen items (chapter 15, figure 15.3).

Reads the recorded verdicts, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_judge.py OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import judge, judge_report
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
GREY, BLUE, RED = "#9CA3AF", "#2563EB", "#E03131"


def main(out: str) -> None:
    items = judge.load_items(ROOT / "datasets" / "judge_items_v1.jsonl")
    first = judge_report.load_rows([ROOT / "runs" / "judge-v1"])
    second = judge_report.load_rows([ROOT / "runs" / "judge-v2-test"])
    one = judge_report._tally(items, first, "unseen")
    two = judge_report._tally(items, second, "unseen")
    rows = [
        (
            "planted faults found",
            ["with praise added", "with fact added", "with capability added"],
            True,
        ),
        ("clean answers flagged", ["planted originals, clean"], False),
        ("supported answers flagged", ["real, my reading: supported"], False),
        ("borderline answers flagged", ["real, my reading: borderline"], None),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=200)
    for i, (label, keys, good) in enumerate(reversed(rows)):
        for shift, counts, color in ((-0.14, one, GREY), (0.14, two, BLUE)):
            k = sum(counts[key]["said"] for key in keys)
            n = sum(counts[key]["n"] for key in keys)
            low, high = wilson_interval(k, n)
            y = i + shift
            ax.plot(
                [100 * low, 100 * high],
                [y, y],
                color=color,
                lw=5,
                solid_capstyle="butt",
            )
            ax.plot([100 * k / n], [y], "o", color=INK, ms=5, zorder=3)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9, color=INK)
    ax.set_xlim(0, 105)
    ax.set_xlabel(
        "Share of verdicts saying unsupported (%, 95% interval)", fontsize=9, color=INK
    )
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Line2D([0], [0], color=c, lw=5) for c in (GREY, BLUE)]
    ax.legend(
        handles,
        ["prompt v1", "prompt v2"],
        fontsize=8,
        frameon=False,
        loc="upper left",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
