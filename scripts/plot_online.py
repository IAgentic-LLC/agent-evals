"""Plot the data figures of chapter 24 from the recorded runs.

`mix`: the same product's met share on two sets of tickets, overall and by slice.
`monitors`: false alarms when nothing changed, and the share of streams that a monitor
caught within a week of a 10-point fall, against how many tickets a day there are.

Usage: uv run --with matplotlib python scripts/plot_online.py mix OUT.png
       uv run --with matplotlib python scripts/plot_online.py monitors OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import online, reliability
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED, AMBER = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626", "#F59E0B"
COLORS = {"threshold": RED, "shewhart": BLUE, "cusum": AMBER}
LABELS = {
    "threshold": "fixed threshold (10 points)",
    "shewhart": "control chart (3 sigma)",
    "cusum": "CUSUM",
}
TRAFFIC = (10, 20, 40, 80, 160)


def _style(ax) -> None:
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def mix(out: str) -> None:
    held = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")}
    chg = {c.case_id: c for c in load_cases(ROOT / "datasets/triage_change_v1.jsonl")}
    sets = [
        (held, read_traces(ROOT / "runs/triage-metered-3-6/traces.jsonl")),
        (chg, read_traces(ROOT / "runs/triage-change-base/traces.jsonl")),
    ]

    def share(cases, traces, key=None, value=None):
        rows = [
            t
            for t in traces
            if key is None or cases[t.case_id].slices.get(key) == value
        ]
        return (
            100
            * sum(reliability.success(cases[t.case_id], t) for t in rows)
            / len(rows)
        )

    groups = [
        ("all tickets", None, None),
        ("technical", "specialist", "technical"),
        ("crash", "runbook_topic", "crash"),
        ("login", "runbook_topic", "login"),
        ("other", "runbook_topic", "other"),
    ]
    fig, ax = plt.subplots(figsize=(5.6, 3.3), dpi=200)
    width = 0.36
    for i, ((cases, traces), color, label) in enumerate(
        zip(sets, (GREY, BLUE), ("held-out tickets", "change-set tickets"), strict=True)
    ):
        xs = [k + (i - 0.5) * width for k in range(len(groups))]
        vals = [share(cases, traces, k, v) for _, k, v in groups]
        ax.bar(xs, vals, width, color=color, label=label)
        for x, v in zip(xs, vals, strict=True):
            ax.text(x, v + 1.5, f"{v:.0f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel("runs meeting the required actions (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 122)
    ax.legend(loc="upper center", fontsize=8.5, frameon=False, ncol=2)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def monitors(out: str) -> None:
    pops = online.load_defaults(ROOT)
    fa = {m: [] for m in online.MONITORS}
    hit = {m: [] for m in online.MONITORS}
    for n in TRAFFIC:
        rates = online.false_alarms(pops["3.6-flash (12)"], n, 500, seed=1)
        found = online.detection(pops["2.5-flash"], pops["3.5-lite"], n, 500, seed=2)
        for m in online.MONITORS:
            fa[m].append(100 * rates[m])
            hit[m].append(100 * online._within(found[m], 7))
    fig, axes = plt.subplots(1, 2, figsize=(6.2, 3.1), dpi=200, sharey=True)
    for ax, data, title in (
        (axes[0], fa, "nothing changed:\nalarms in 30 days"),
        (axes[1], hit, "a 10-point fall:\ncaught within 7 days"),
    ):
        for m in online.MONITORS:
            ax.plot(
                TRAFFIC, data[m], "o-", color=COLORS[m], lw=2, ms=4, label=LABELS[m]
            )
        ax.set_xscale("log")
        ax.set_xticks(TRAFFIC)
        ax.set_xticklabels([str(n) for n in TRAFFIC])
        ax.set_xlabel("tickets a day (log scale)", fontsize=9, color=INK)
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_ylim(-4, 104)
        _style(ax)
    axes[0].set_ylabel("share of streams (%)", fontsize=9, color=INK)
    axes[0].legend(loc="center left", fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"mix": mix, "monitors": monitors}[sys.argv[1]](sys.argv[2])
