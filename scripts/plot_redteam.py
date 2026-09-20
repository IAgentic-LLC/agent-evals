"""Plot tickets hit by an attack, by round and door, with and without the note (chapter 21).

Usage: uv run --with matplotlib python scripts/plot_redteam.py OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import redteam
from agent_evals.runner import load_cases, read_traces
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE = "#1F2937", "#9CA3AF", "#2563EB"


def ever_hit(cases, traces, keep):
    by_ticket: dict[str, bool] = {}
    for t in traces:
        c = cases[t.case_id]
        if keep(c):
            by_ticket[t.case_id] = by_ticket.get(t.case_id, False) or redteam.hit(c, t)
    return sum(by_ticket.values()), len(by_ticket)


def load(name):
    return read_traces(ROOT / "runs" / name / "traces.jsonl")


def main(out: str) -> None:
    one = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_redteam_v1.jsonl")}
    two = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_redteam_v2.jsonl")}
    groups = [
        (
            "round 1\nin the ticket text",
            one,
            (load("triage-redteam-1"), load("triage-redteam-1-untrusted")),
            lambda c: c.slices["wall"] == "in_scope",
        ),
        (
            "round 2\nprocedure note\nin the ticket",
            two,
            (load("triage-redteam-2"), load("triage-redteam-2-untrusted")),
            lambda c: c.slices["door"] == "ticket",
        ),
        (
            "round 2\npoisoned runbook\nreply",
            two,
            (load("triage-redteam-2"), load("triage-redteam-2-untrusted")),
            lambda c: c.slices["door"] == "runbook",
        ),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    width = 0.36
    for i, (label, cases, runs, keep) in enumerate(groups):
        for j, (traces, color, name) in enumerate(
            zip(runs, (GREY, BLUE), ("as shipped", "with the note"))
        ):
            k, n = ever_hit(cases, traces, keep)
            low, high = wilson_interval(k, n)
            x = i + (j - 0.5) * width
            ax.bar(
                x,
                100 * k / n,
                width,
                color=color,
                label=name if i == 0 else None,
            )
            ax.plot([x, x], [100 * low, 100 * high], color=INK, lw=1.2)
            ax.text(
                x,
                100 * high + 2,
                f"{k}/{n}",
                ha="center",
                fontsize=8,
                color=INK,
            )
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8.5, color=INK)
    ax.set_ylabel("tickets where the attack worked (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.tick_params(axis="y", labelsize=9, colors=INK)
    ax.legend(loc="upper left", fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
