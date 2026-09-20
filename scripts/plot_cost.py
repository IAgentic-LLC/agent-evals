"""Plot the data figures of chapter 22 from the recorded runs.

`frontier`: cost per run against the share of runs that met the required actions.
`latency`: how long runs took, as the share of runs finished by a given time.
`thinking`: what a run costs, split into the tokens the model wrote and the ones it
thought and did not show.

Usage: uv run --with matplotlib python scripts/plot_cost.py frontier OUT.png
       uv run --with matplotlib python scripts/plot_cost.py latency OUT.png
       uv run --with matplotlib python scripts/plot_cost.py thinking OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

from agent_evals import cost
from agent_evals.cli import COST_RUNS
from agent_evals.runner import load_cases, read_traces
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED, AMBER = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626", "#F59E0B"
COLORS = {
    "3.6-flash": GREY,
    "3.5-flash-lite": AMBER,
    "2.5-flash": BLUE,
    "3.6 + stall guard": RED,
}


def _load():
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    prices = cost.load_prices(ROOT / "config/prices.yaml")
    conditions = [
        (label, model, read_traces(ROOT / "runs" / run / "traces.jsonl"))
        for label, model, run in COST_RUNS
    ]
    return cases, prices, conditions


def frontier(out: str) -> None:
    cases, prices, conditions = _load()
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    points = []
    for label, model, traces in conditions:
        n = len(traces)
        met = sum(cost.outcome(cases[t.case_id], t) == "met" for t in traces)
        per_run = sum(cost.run_cost(t, prices[model]) for t in traces) / n
        low, high = wilson_interval(met, n)
        points.append((label, per_run, met / n))
        ax.plot([per_run, per_run], [100 * low, 100 * high], color=COLORS[label], lw=2)
        ax.plot([per_run], [100 * met / n], "o", color=COLORS[label], ms=7)
    front = set(cost.pareto(points))
    for label, per_run, share in points:
        if label in front:
            ax.plot(
                [per_run],
                [100 * share],
                "o",
                mfc="none",
                mec=INK,
                mew=1.5,
                ms=13,
            )
        left = label == "3.5-flash-lite"
        ax.annotate(
            label,
            (per_run, 100 * share),
            textcoords="offset points",
            xytext=(-10 if left else 8, -3),
            ha="right" if left else "left",
            fontsize=8.5,
            color=INK,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.00032, 0.011)
    ax.set_ylim(40, 100)
    ax.set_xlabel("dollars per run (log scale)", fontsize=9, color=INK)
    ax.set_ylabel("runs meeting the required actions (%)", fontsize=9, color=INK)
    ax.set_xticks([0.0007, 0.0012, 0.004])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_xticklabels(["$0.0007", "$0.0012", "$0.0040"])
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def latency(out: str) -> None:
    _, _, conditions = _load()
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    for label, _, traces in conditions:
        values = sorted(t.latency_s for t in traces if t.latency_s is not None)
        share = [100 * (i + 1) / len(values) for i in range(len(values))]
        ax.step(values, share, where="post", color=COLORS[label], lw=2, label=label)
    ax.axhline(95, color=GREY, lw=0.8, ls="--")
    ax.text(0.9, 93, "95th percentile", fontsize=8, color=INK, va="top")
    ax.set_xscale("log")
    ax.set_xlim(0.8, 60)
    ax.set_xticks([1, 2, 5, 10, 20, 50])
    ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
    ax.set_xlabel("seconds for a run (log scale)", fontsize=9, color=INK)
    ax.set_ylabel("runs finished by then (%)", fontsize=9, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def thinking(out: str) -> None:
    _, prices, conditions = _load()
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=200)
    labels = [c[0] for c in conditions][::-1]
    written, unseen = [], []
    for label, model, traces in conditions[::-1]:
        n = len(traces)
        thought = sum(t.usage["thinking_tokens"] for t in traces) / n
        hidden = thought * prices[model][1] / 1_000_000
        total = sum(cost.run_cost(t, prices[model]) for t in traces) / n
        written.append(total - hidden)
        unseen.append(hidden)
    ax.barh(labels, written, color=GREY, label="input and written tokens")
    ax.barh(
        labels, unseen, left=written, color=RED, label="thinking tokens, never shown"
    )
    for i, (w, u) in enumerate(zip(written, unseen, strict=True)):
        ax.text(
            w + u + 0.00008, i, f"${w + u:.4f}", va="center", fontsize=8.5, color=INK
        )
    ax.set_xlim(0, 0.0056)
    ax.set_xlabel("dollars per run", fontsize=9, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    ax.legend(loc="center right", fontsize=8.5, frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"frontier": frontier, "latency": latency, "thinking": thinking}[sys.argv[1]](
        sys.argv[2]
    )
