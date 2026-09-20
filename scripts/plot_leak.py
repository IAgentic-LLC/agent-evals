"""Plot required-action passes for three clean runs and the leaky run (chapter 7, figure 7.2).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_leak.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.graders import required_actions_missing
from agent_evals.runner import load_cases, read_traces
from agent_evals.scorecard import build_scorecard
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
BLUE, RED, INK = "#2563EB", "#E03131", "#1F2937"


def main(out: str) -> None:
    cases = load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    by_id = {c.case_id: c for c in cases}
    rows = []
    for label, name in (
        ("Clean run, pass 1", "triage-heldout-v1"),
        ("Clean run, pass 2", "triage-heldout-v1-2"),
        ("Clean run, pass 3", "triage-heldout-v1-3"),
    ):
        card = build_scorecard(
            "t", name, cases, read_traces(ROOT / "runs" / name / "traces.jsonl")
        )
        rows.append((label, card.actions_successes, BLUE))
    leaky = read_traces(ROOT / "runs/triage-heldout-v1-leaky/traces.jsonl")
    card = build_scorecard("t", "leaky", cases, leaky)
    rows.append(("Leaky run, as scored", card.actions_successes, RED))
    own = 0
    for t in leaky:
        mine = t.model_copy(
            update={"actions_taken": t.actions_taken[t.ledger_at_start :]}
        )
        own += t.error is None and not required_actions_missing(by_id[t.case_id], mine)
    rows.append(("Leaky run, own actions only", own, BLUE))

    n = len(cases)
    fig, ax = plt.subplots(figsize=(5.5, 3.1), dpi=200)
    for i, (label, k, color) in enumerate(reversed(rows)):
        low, high = wilson_interval(k, n)
        ax.plot(
            [100 * low, 100 * high],
            [i, i],
            color=color,
            linewidth=7,
            solid_capstyle="butt",
        )
        ax.plot([100 * k / n], [i], marker="o", color=INK, markersize=6, zorder=3)
        ax.text(100 * low, i + 0.2, f"{k} of {n}", va="bottom", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10, color=INK)
    ax.set_xlim(30, 100)
    ax.set_ylim(-0.6, len(rows) - 0.3)
    ax.set_xlabel(
        "Required actions met (95% interval, percent)",
        fontsize=10,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
