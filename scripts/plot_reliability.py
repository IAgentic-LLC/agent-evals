"""Plot the two data figures of chapter 19 from the recorded runs.

`curve`: pass@k and pass^k against k, with what independence would predict.
`grid`: every case by every trial, sorted by how often the case succeeded.

Usage: uv run --with matplotlib python scripts/plot_reliability.py curve OUT.png
       uv run --with matplotlib python scripts/plot_reliability.py grid OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from agent_evals import compare, reliability
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626"


def _outcomes():
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    names = [n for n in reliability.TRIAL_RUNS if (ROOT / "runs" / n).is_dir()]
    runs = [read_traces(ROOT / "runs" / n / "traces.jsonl") for n in names]
    return reliability.outcomes(cases, runs)


def curve(out: str) -> None:
    outs = _outcomes()
    n = len(next(iter(outs.values())))
    ks = list(range(1, n + 1))
    rows = reliability.curve(outs, range(1, n + 1))
    p1 = compare.mean(reliability.per_case(outs, 1, reliability.pass_hat))
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    ax.plot(ks, [100 * r[1] for r in rows], "o-", color=BLUE, lw=2, label="pass@k")
    ax.plot(ks, [100 * r[2] for r in rows], "o-", color=INK, lw=2, label="pass^k")
    ax.plot(
        ks,
        [100 * p1**k for k in ks],
        "--",
        color=RED,
        lw=1.5,
        label="pass^k if runs were independent",
    )
    ax.set_xlabel("k, runs per case", fontsize=9, color=INK)
    ax.set_ylabel("cases meeting the required actions (%)", fontsize=9, color=INK)
    ax.set_ylim(0, 100)
    ax.set_xticks(ks)
    ax.tick_params(labelsize=9, colors=INK)
    ax.legend(fontsize=8.5, frameon=False, loc="center right")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def grid(out: str) -> None:
    outs = _outcomes()
    n = len(next(iter(outs.values())))
    order = sorted(outs, key=lambda c: (-sum(outs[c]), c))
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    data = [[1 if x else 0 for x in outs[c]] for c in order]
    ax.imshow(
        list(map(list, zip(*data))),
        aspect="auto",
        cmap=ListedColormap(["#FECACA", BLUE]),
        interpolation="nearest",
    )
    ax.set_xlabel(
        f"the {len(order)} cases, sorted by how often they succeeded",
        fontsize=9,
        color=INK,
    )
    ax.set_ylabel("trial", fontsize=9, color=INK)
    ax.set_yticks(range(n))
    ax.set_yticklabels([str(i + 1) for i in range(n)])
    ax.set_xticks([])
    ax.tick_params(labelsize=9, colors=INK)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"curve": curve, "grid": grid}[sys.argv[1]](sys.argv[2])
