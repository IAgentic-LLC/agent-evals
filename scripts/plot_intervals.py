"""Plot what a perfect run can show at different sample sizes (chapter 1, figure 1.1).

Usage: uv run --with matplotlib python scripts/plot_intervals.py OUTPUT.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.stats import wilson_interval

BLUE = "#2563EB"
INK = "#1F2937"
SIZES = [6, 20, 50, 100]


def main(out: str) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=200)
    for i, n in enumerate(reversed(SIZES)):
        low, high = wilson_interval(n, n)
        ax.plot(
            [100 * low, 100 * high],
            [i, i],
            color=BLUE,
            linewidth=7,
            solid_capstyle="butt",
        )
        ax.plot([100], [i], marker="o", color=INK, markersize=6, zorder=3)
        ax.text(
            100 * low,
            i + 0.2,
            f"{100 * low:.1f}%",
            ha="left",
            va="bottom",
            fontsize=9,
            color=INK,
        )
    ax.axvline(80, color=INK, linestyle="--", linewidth=1)
    ax.text(80.4, 3.55, "80% target", fontsize=8.5, color=INK, va="bottom")
    ax.set_yticks(range(len(SIZES)))
    ax.set_yticklabels(
        [f"{n} of {n}" for n in reversed(SIZES)], fontsize=9.5, color=INK
    )
    ax.set_xlim(50, 101)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlabel(
        "Success rates the result cannot rule out (95% Wilson interval)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
