"""Plot the data figures of chapter 25 from the recorded runs.

`fix`: for each version of the product on the incident set, the share of runs that tried
another customer's id and the share that acted on it, and the share that met the required
actions.
`reach`: the chance that a check of a given size sees the incident at least once, for the
incident's own ticket, and for a general held-out set where that ticket is one in 42.

Usage: uv run --with matplotlib python scripts/plot_incident.py fix OUT.png
       uv run --with matplotlib python scripts/plot_incident.py reach OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import cli, incident, reliability
from agent_evals.answer_graders import acted_on_the_right_customer
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED, AMBER = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626", "#F59E0B"


def _style(ax) -> None:
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def fix(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_incident_v1.jsonl")
    }
    labels, tried, acted, met = [], [], [], []
    for label, name in cli.INCIDENT_FIXES[:4]:
        rows = read_traces(ROOT / "runs" / name / "traces.jsonl")
        n = len(rows)
        labels.append(label.replace(" and ", " +\n"))
        tried.append(
            100
            * sum(incident.attempted_other_customer(cases[t.case_id], t) for t in rows)
            / n
        )
        acted.append(
            100
            * sum(not acted_on_the_right_customer(cases[t.case_id], t) for t in rows)
            / n
        )
        met.append(
            100 * sum(reliability.success(cases[t.case_id], t) for t in rows) / n
        )
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2), dpi=200)
    width = 0.36
    xs = range(len(labels))
    for i, (vals, color, name) in enumerate(
        ((tried, AMBER, "tried another id"), (acted, RED, "acted on it"))
    ):
        pos = [x + (i - 0.5) * width for x in xs]
        axes[0].bar(pos, vals, width, color=color, label=name)
        for x, v in zip(pos, vals, strict=True):
            axes[0].text(x, v + 0.15, f"{v:.1f}", ha="center", fontsize=7.5, color=INK)
    axes[0].set_ylabel("runs of 480 (%)", fontsize=9, color=INK)
    axes[0].set_ylim(0, 7)
    axes[0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[1].bar(list(xs), met, 0.55, color=BLUE)
    for x, v in zip(xs, met, strict=True):
        axes[1].text(x, v + 1.2, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    axes[1].set_ylabel("runs meeting the required actions (%)", fontsize=9, color=INK)
    axes[1].set_ylim(0, 115)
    for ax in axes:
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, fontsize=7.5)
        _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def reach(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_incident_v1.jsonl")
    }
    rows = [
        t
        for t in read_traces(ROOT / "runs/triage-incident-lite/traces.jsonl")
        if t.case_id == "IN-001"
    ]
    p_own = sum(
        not acted_on_the_right_customer(cases["IN-001"], t) for t in rows
    ) / len(rows)
    p_held = p_own / 42
    sizes = list(range(1, 301))
    fig, ax = plt.subplots(figsize=(5.8, 3.2), dpi=200)
    for p, color, label in (
        (p_own, BLUE, "the incident's own ticket"),
        (p_held, GREY, "a held-out set (1 ticket in 42)"),
    ):
        ax.plot(
            sizes,
            [100 * (1 - (1 - p) ** n) for n in sizes],
            color=color,
            lw=2,
            label=label,
        )
    for n, p, color in ((3, p_own, BLUE), (126, p_held, GREY)):
        ax.plot([n], [100 * (1 - (1 - p) ** n)], "o", color=color, ms=5)
        ax.text(
            n + 4,
            100 * (1 - (1 - p) ** n) - 7,
            f"{100 * (1 - (1 - p) ** n):.0f}% at {n} runs",
            fontsize=8,
            color=INK,
        )
    ax.set_xlabel("runs in the check", fontsize=9, color=INK)
    ax.set_ylabel("chance of seeing the incident (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"fix": fix, "reach": reach}[sys.argv[1]](sys.argv[2])
