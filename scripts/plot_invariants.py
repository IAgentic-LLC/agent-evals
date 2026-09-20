"""Plot how many tickets broke a protected invariant under three policies (figure 8.2).

Reads the recorded runs, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_invariants.py OUTPUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import invariants
from agent_evals.runner import load_cases, read_traces
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
POLICY_COLORS = {
    "deny-list": "#9CA3AF",
    "required-only": "#E03131",
    "permitted": "#2563EB",
}
RUNS = (
    ("Fixed product, pass 1", "triage-heldout-v1"),
    ("Fixed product, pass 2", "triage-heldout-v1-2"),
    ("Fixed product, pass 3", "triage-heldout-v1-3"),
    ("Product as shipped", "triage-heldout-v1-shipped"),
)


def main(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    permissions = invariants.load_permissions(
        ROOT / "datasets/triage_heldout_v1.permissions.jsonl"
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=200)
    y, ticks, labels = 0.0, [], []
    for label, run in reversed(RUNS):
        traces = read_traces(ROOT / "runs" / run / "traces.jsonl")
        first = y
        for policy, color in reversed(POLICY_COLORS.items()):
            k = sum(
                bool(invariants.violations(cases[t.case_id], t, policy, permissions))
                for t in traces
            )
            low, high = wilson_interval(k, len(traces))
            ax.plot(
                [100 * low, 100 * high],
                [y, y],
                color=color,
                lw=6,
                solid_capstyle="butt",
            )
            ax.plot([100 * k / len(traces)], [y], "o", color=INK, ms=5, zorder=3)
            ax.text(
                100 * high + 1.2,
                y,
                f"{k} of {len(traces)}",
                va="center",
                fontsize=9,
                color=INK,
            )
            y += 1
        ticks.append((first + y - 1) / 2)
        labels.append(label)
        y += 0.8
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels, fontsize=10, color=INK)
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.7, y - 0.3)
    ax.set_xlabel(
        "Tickets that broke an invariant (95% interval, percent)",
        fontsize=10,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=10, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [
        plt.Line2D([0], [0], color=c, lw=6, label=p) for p, c in POLICY_COLORS.items()
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
