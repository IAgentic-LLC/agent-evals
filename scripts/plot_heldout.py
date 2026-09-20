"""Plot routing on the development six against the held-out set, by specialist (chapter 4).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_heldout.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals.runner import load_cases, read_traces
from agent_evals.scorecard import build_scorecard

ROOT = Path(__file__).resolve().parents[1]
BLUE = "#2563EB"
RED = "#E03131"
INK = "#1F2937"


def _card(cases, traces):
    return build_scorecard("", "", cases, traces)


def main(out: str) -> None:
    six_cases = load_cases(ROOT / "datasets/triage_book3_six.jsonl")
    six = read_traces(ROOT / "runs/triage-dev-six-ch04/traces.jsonl")
    test_cases = load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    test = read_traces(ROOT / "runs/triage-heldout-v1/traces.jsonl")
    by_id = {c.case_id: c for c in test_cases}

    rows = [("Development six", _card(six_cases, six), BLUE)]
    for name in ("billing", "security", "technical"):
        group = [t for t in test if by_id[t.case_id].expected["handled_by"] == name]
        color = RED if name == "technical" else BLUE
        rows.append((f"Test set: {name}", _card(test_cases, group), color))

    fig, ax = plt.subplots(figsize=(5.5, 2.9), dpi=200)
    for i, (_, card, color) in enumerate(reversed(rows)):
        low, high = 100 * card.routing_interval.low, 100 * card.routing_interval.high
        n, k = card.observations, card.routing_successes
        ax.plot([low, high], [i, i], color=color, linewidth=7, solid_capstyle="butt")
        ax.plot([100 * k / n], [i], marker="o", color=INK, markersize=6, zorder=3)
        ax.text(
            low,
            i + 0.2,
            f"{k} of {n}",
            ha="left",
            va="bottom",
            fontsize=10.5,
            color=INK,
        )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=10.5, color=INK)
    ax.set_xlim(0, 101)
    ax.set_ylim(-0.6, 3.9)
    ax.set_xlabel(
        "Tickets routed correctly (95% interval, percent)", fontsize=10, color=INK
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
