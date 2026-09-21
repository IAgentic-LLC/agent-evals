"""Plot the data figures of chapter 23 from the recorded runs.

`characteristics`: for each gate rule, the share of draws that did not pass, against how
much worse the candidate really was. Nothing changed is at zero.
`change`: the met share by runbook topic, the change against the product as shipped.

Usage: uv run --with matplotlib python scripts/plot_gates.py characteristics OUT.png
       uv run --with matplotlib python scripts/plot_gates.py change OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import gate_study, reliability
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED, AMBER = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626", "#F59E0B"
COLORS = {
    "point, drop over 5": RED,
    "sign test": AMBER,
    "interval, margin 10": BLUE,
    "interval, margin 5": GREY,
}


def _style(ax) -> None:
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def characteristics(out: str) -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    data = gate_study.load(ROOT, cases)
    nothing = gate_study.pooled([gate_study.nothing_changed(v) for v in data.values()])
    points: dict[str, list[tuple[float, float]]] = {
        label: [(0.0, 100 * (nothing[label].block + nothing[label].hold))]
        for label, _, _ in gate_study.RULES
    }
    for base, cand in gate_study.PAIRS:
        drop = gate_study.measured_drop(data[base], data[cand])
        outcomes = gate_study.real_drop(data[base], data[cand])
        for label, _, _ in gate_study.RULES:
            o = outcomes[label]
            points[label].append((drop, 100 * (o.block + o.hold)))
    fig, ax = plt.subplots(figsize=(5.6, 3.6), dpi=200)
    for label, pts in points.items():
        pts.sort()
        ax.plot(
            [p[0] for p in pts],
            [p[1] for p in pts],
            "o-",
            color=COLORS[label],
            lw=2,
            ms=4.5,
            label=label,
        )
    ax.axvline(0, color=GREY, lw=0.8, ls="--")
    ax.set_xlabel(
        "how much worse the candidate really was (points; below 0 it was better)",
        fontsize=9,
        color=INK,
    )
    ax.set_ylabel("draws that did not pass (%)", fontsize=9, color=INK)
    ax.set_ylim(-4, 104)
    ax.text(0.6, 52, "nothing\nchanged", fontsize=8, color=INK, va="center")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def change(out: str) -> None:
    cases = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_change_v1.jsonl")}
    runs = [
        read_traces(ROOT / "runs" / name / "traces.jsonl")
        for name in ("triage-change-base", "triage-change-topics")
    ]
    topics = ["crash", "login", "other", "all"]

    def share(traces, topic):
        rows = [
            t
            for t in traces
            if topic == "all" or cases[t.case_id].slices["runbook_topic"] == topic
        ]
        return (
            100
            * sum(reliability.success(cases[t.case_id], t) for t in rows)
            / len(rows)
        )

    fig, ax = plt.subplots(figsize=(5.6, 3.2), dpi=200)
    width = 0.36
    for i, (traces, color, label) in enumerate(
        zip(runs, (GREY, BLUE), ("as shipped", "with the topics hint"), strict=True)
    ):
        xs = [k + (i - 0.5) * width for k in range(len(topics))]
        vals = [share(traces, t) for t in topics]
        ax.bar(xs, vals, width, color=color, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(topics)))
    ax.set_xticklabels(["crash (12)", "login (10)", "other (18)", "all (40)"])
    ax.set_ylabel("runs meeting the required actions (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 122)
    ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"characteristics": characteristics, "change": change}[sys.argv[1]](sys.argv[2])
