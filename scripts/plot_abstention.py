"""Plot the top similarity score for each kind of question (chapter 14, figure 14.2).

Reads the recorded run and the dataset, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_abstention.py SCORES.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import abstention
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
BLUE, GREY, RED, AMBER = "#2563EB", "#9CA3AF", "#E03131", "#F08C00"


def main(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets" / "pkg_abstain_v1.jsonl")
    }
    traces = read_traces(ROOT / "runs" / "pkg-abstain-1" / "traces.jsonl")
    rows: dict[str, list[float]] = {}
    for t in traces:
        row = abstention.group(cases[t.case_id], t)
        rows.setdefault(row, []).append(float(t.retrieved[0]["score"]))
    order = [
        "should answer",
        "outside",
        "beyond_summary",
        "false_premise",
        "none",
        "fresh",
        "retrieval miss",
    ]
    colors = {"should answer": BLUE, "retrieval miss": AMBER}
    fig, ax = plt.subplots(figsize=(5.5, 3.3), dpi=200)
    for i, name in enumerate(reversed(order)):
        values = rows[name]
        jitter = [i + (k % 5 - 2) * 0.06 for k in range(len(values))]
        ax.scatter(
            values,
            jitter,
            s=20,
            color=colors.get(name, RED),
            alpha=0.85,
            edgecolors="none",
        )
    cutoff = abstention.choose_cutoff(cases, traces)
    for x, label in ((0.60, "0.60"), (cutoff, f"{cutoff:.3f}")):
        ax.axvline(x, color=GREY, lw=1, ls="--")
        ax.text(x, len(order) - 0.35, label, ha="center", fontsize=8, color=INK)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=9, color=INK)
    ax.set_xlabel("Cosine similarity of the best match", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
