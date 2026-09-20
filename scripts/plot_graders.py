"""Plot each grader version's errors on the dev and test answers (chapter 5, figure 5.2).

Reads the labels and the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_graders.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.answer_graders import ASKS_FOR_KNOWN_INFO
from agent_evals.grader_check import check, load_cases_by_id, load_labels
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563EB"
TEAL = "#0D9488"
INK = "#1F2937"


def main(out: str) -> None:
    labels = load_labels(ROOT / "datasets/graders/asks_for_known_info.labels.jsonl")
    cases = load_cases_by_id(
        ROOT / "datasets/triage_book3_six.jsonl",
        ROOT / "datasets/triage_heldout_v1.jsonl",
    )
    fig, ax = plt.subplots(figsize=(5.5, 3.3), dpi=200)
    versions = list(ASKS_FOR_KNOWN_INFO)
    for i, version in enumerate(reversed(versions)):
        for offset, split, color in ((0.17, "dev", BLUE), (-0.17, "test", TEAL)):
            part = [r for r in labels if r["split"] == split]
            r = check(ASKS_FOR_KNOWN_INFO[version], part, ROOT / "runs", cases)
            errors = r.fp + r.fn
            low, high = wilson_interval(errors, r.total)
            y = i + offset
            ax.plot([100 * low, 100 * high], [y, y], color=color, linewidth=5)
            ax.plot(
                [100 * errors / r.total],
                [y],
                marker="o",
                color=INK,
                markersize=4.5,
                zorder=3,
            )
            ax.text(
                100 * high + 1, y, f"{errors}", va="center", fontsize=9.5, color=INK
            )
    ax.set_yticks(range(len(versions)))
    ax.set_yticklabels(list(reversed(versions)), fontsize=10.5, color=INK)
    ax.set_xlim(0, 32)
    ax.set_xlabel(
        "Answers graded wrongly (95% interval, percent)", fontsize=10, color=INK
    )
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.plot([], [], color=BLUE, linewidth=5, label="dev answers (117)")
    ax.plot([], [], color=TEAL, linewidth=5, label="test answers (59)")
    ax.legend(frameon=False, fontsize=9.5, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
