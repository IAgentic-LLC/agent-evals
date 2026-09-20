"""Plot how the runs of chapter 20 ended, by kind of ticket, from the recorded runs.

Usage: uv run --with matplotlib python scripts/plot_routing.py OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import reliability, routing
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
BLUE, RED, AMBER = "#2563EB", "#DC2626", "#F59E0B"


def _ends(cases, traces, keep):
    counts = {"finished right": 0, "handoff loop": 0, "tool loop": 0}
    n = 0
    for t in traces:
        if not keep(cases[t.case_id]):
            continue
        n += 1
        if t.error is None and routing.where(cases[t.case_id], t) == "right":
            counts["finished right"] += 1
        elif t.error and t.error.startswith("HandoffLoop"):
            counts["handoff loop"] += 1
        else:
            counts["tool loop"] += 1
    return counts, n


def main(out: str) -> None:
    held = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")}
    held_runs = [
        t
        for n in reliability.TRIAL_RUNS
        for t in read_traces(ROOT / "runs" / n / "traces.jsonl")
    ]
    rset = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_routing_v1.jsonl")}
    rruns = read_traces(ROOT / "runs/triage-routing-3x/traces.jsonl")
    rows = []
    for name in ("billing", "security", "technical"):
        counts, n = _ends(
            held, held_runs, lambda c, name=name: c.expected["handled_by"] == name
        )
        rows.append((f"42 tickets: {name}", counts, n))
    for kind in ("misfiled", "boundary", "distractor"):
        counts, n = _ends(rset, rruns, lambda c, kind=kind: c.slices["kind"] == kind)
        rows.append((f"routing set: {kind}", counts, n))
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    colors = (BLUE, RED, AMBER)
    for i, (label, counts, n) in enumerate(reversed(rows)):
        left = 0.0
        for (name, k), color in zip(counts.items(), colors):
            share = 100 * k / n
            ax.barh(
                i,
                share,
                left=left,
                color=color,
                height=0.62,
                label=name if i == 0 else None,
            )
            left += share
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(
        [f"{r[0]} ({r[2]} runs)" for r in reversed(rows)], fontsize=8.5, color=INK
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("share of runs (%)", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.2, 1.0), ncol=3, fontsize=8, frameon=False
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
