"""Plot what each runbook search returned, run by run (chapter 9, figure 9.2).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_searches.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.runner import load_cases, read_traces
from agent_evals.tool_calls import OUTCOMES, search_outcome

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
COLORS = {"relevant": "#2563EB", "irrelevant": "#F08C00", "empty": "#9CA3AF"}
RUNS = (
    ("Unchanged tool, pass 1", "triage-heldout-v1-calls"),
    ("Unchanged tool, pass 2", "triage-heldout-v1-calls-2"),
    ("Topics hint, pass 1", "triage-heldout-v1-topics"),
    ("Topics hint, pass 2", "triage-heldout-v1-topics-2"),
)


def main(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=200)
    for row, (label, run) in enumerate(reversed(RUNS)):
        counts = dict.fromkeys(OUTCOMES, 0)
        for t in read_traces(ROOT / "runs" / run / "traces.jsonl"):
            for call in t.tool_calls:
                kind = search_outcome(cases[t.case_id], call)
                if kind:
                    counts[kind] += 1
        left = 0
        for kind in OUTCOMES:
            n = counts[kind]
            ax.barh(row, n, left=left, color=COLORS[kind], height=0.62)
            if n >= 4:
                ax.text(
                    left + n / 2,
                    row,
                    str(n),
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white",
                )
            left += n
    ax.set_yticks(range(len(RUNS)))
    ax.set_yticklabels([r[0] for r in reversed(RUNS)], fontsize=10, color=INK)
    ax.set_xlabel("Runbook searches (42 tickets per run)", fontsize=10, color=INK)
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[k]) for k in OUTCOMES]
    ax.legend(
        handles,
        OUTCOMES,
        loc="lower right",
        fontsize=9,
        frameon=False,
        ncol=3,
        bbox_to_anchor=(1, -0.42),
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white", bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
