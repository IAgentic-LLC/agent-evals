"""Plot how much a judge's kappa moves when two of my labels change (chapter 17, figure 17.2).

Reads the recorded verdicts and my readings, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_agreement.py OUT.png
"""

import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import agreement, labels

INK = "#1F2937"
GREY, BLUE = "#9CA3AF", "#2563EB"
RUNS = (
    "judge-v2-all",
    "judge-v3-all",
    "judge-v3u-all",
    "judge-v2-self",
    "judge-v2-other",
)


def kappa(items, truth, run):
    real = [i for i in items if i["group"] == "real"]
    judged = labels.judge_labels(run)
    a = [truth[i["item_id"]] for i in real]
    b = [judged[i["item_id"]] for i in real]
    low, high = agreement.bootstrap_interval(
        agreement.cohen_kappa, a, b, groups=[i["question"] for i in real]
    )
    return agreement.cohen_kappa(a, b), low, high


def main(out: str) -> None:
    items = labels.item_list()
    before = labels.author_labels(items)
    after = labels.author_labels(items, adjudicated=True)
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    for i, run in enumerate(reversed(RUNS)):
        for shift, truth, color in ((-0.16, before, GREY), (0.16, after, BLUE)):
            k, low, high = kappa(items, truth, run)
            y = i + shift
            ax.plot([low, high], [y, y], color=color, lw=5, solid_capstyle="butt")
            ax.plot([k], [y], "o", color=INK, ms=5, zorder=3)
    ax.axvline(0, color=GREY, lw=0.8)
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels(
        [r.replace("judge-", "") for r in reversed(RUNS)], fontsize=9, color=INK
    )
    ax.set_xlim(-0.2, 1.05)
    ax.set_xlabel(
        "Cohen's kappa with my labels, real answers (95% interval)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Line2D([0], [0], color=c, lw=5) for c in (GREY, BLUE)]
    ax.legend(
        handles,
        ["my readings as first written", "after I changed two"],
        fontsize=8,
        frameon=False,
        loc="lower center",
        bbox_to_anchor=(0.45, 1.0),
        ncol=2,
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
