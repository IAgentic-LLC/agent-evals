"""Plot routing against required actions for the two recorded runs (chapter 2, figure 2.1).

Reads the recorded scorecards, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_success.py OUTPUT.png
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BLUE = "#2563EB"
INK = "#1F2937"
RUNS = Path(__file__).resolve().parents[1] / "runs"


def _card(run: str) -> dict:
    return json.loads((RUNS / run / "scorecard.json").read_text(encoding="utf8"))


def main(out: str) -> None:
    shipped = _card("triage-live-shipped")
    fixed = _card("triage-live-customer-id")
    rows = [
        ("Routing correct", shipped["routing_successes"], shipped["routing_interval"]),
        (
            "Required actions,\nas shipped",
            shipped["actions_successes"],
            shipped["actions_interval"],
        ),
        (
            "Required actions,\nID passed on",
            fixed["actions_successes"],
            fixed["actions_interval"],
        ),
    ]
    n = shipped["observations"]
    fig, ax = plt.subplots(figsize=(5.5, 2.9), dpi=200)
    for i, (_, successes, iv) in enumerate(reversed(rows)):
        low, high = 100 * iv["low"], 100 * iv["high"]
        ax.plot([low, high], [i, i], color=BLUE, linewidth=7, solid_capstyle="butt")
        ax.plot(
            [100 * successes / n], [i], marker="o", color=INK, markersize=6, zorder=3
        )
        ax.text(
            low,
            i + 0.2,
            f"{successes} of {n}",
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=INK,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10.5, color=INK)
    ax.set_xlim(20, 101)
    ax.set_ylim(-0.6, 2.9)
    ax.set_xlabel("Success rate (95% interval, percent)", fontsize=10, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
