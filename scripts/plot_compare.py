"""Plot the two data figures of chapter 18 from the recorded runs.

`intervals`: four intervals for the same small difference between two prompts.
`power`: how often a real gap would be seen, by number of questions.

Usage: uv run --with matplotlib python scripts/plot_compare.py intervals OUT.png
       uv run --with matplotlib python scripts/plot_compare.py power OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import abstention, compare
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626"


def _inputs():
    cases = {c.case_id: c for c in load_cases(ROOT / "datasets/pkg_abstain_v1.jsonl")}
    runs = {
        name: [read_traces(ROOT / "runs" / r / "traces.jsonl") for r in dirs]
        for name, dirs in abstention.CONFIGS.items()
    }
    return cases, runs


def intervals(out: str) -> None:
    cases, runs = _inputs()
    outcomes = compare.question_outcomes(cases, runs)
    names, questions, rates = compare._rates(outcomes)
    first, second = names
    flat = {n: [x for q in questions for x in outcomes[n][q]] for n in names}
    k1, k2 = sum(flat[first]), sum(flat[second])
    n = len(flat[first])
    diff = [y - x for x, y in zip(rates[first], rates[second])]
    rows = [
        (
            "two unrelated groups\nof 110 answers",
            100 * (k2 / n - k1 / n),
            compare.newcombe_difference(k2, n, k1, n),
            GREY,
        ),
        (
            "same questions, redrawn\ntogether (both runs)",
            100 * compare.mean(diff),
            compare.cluster_interval(compare.mean, diff),
            BLUE,
        ),
    ]
    for number in (0, 1):
        a = [outcomes[first][q][number] for q in questions]
        b = [outcomes[second][q][number] for q in questions]
        both, only_first, only_second, neither = compare.paired_counts(a, b)
        low, high = compare.bayes_paired_interval(
            only_second, only_first, both, neither
        )
        rows.append(
            (
                f"paired Bayes,\nrun {number + 1} only",
                100 * (only_second - only_first) / len(questions),
                (low, high),
                BLUE,
            )
        )
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=200)
    for i, (label, point, (low, high), color) in enumerate(reversed(rows)):
        ax.plot(
            [100 * low, 100 * high], [i, i], color=color, lw=6, solid_capstyle="butt"
        )
        ax.plot([point], [i], "o", color=INK, ms=5, zorder=3)
    ax.axvline(0, color=INK, lw=0.8)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=8.5, color=INK)
    ax.set_xlim(-15, 12)
    ax.set_xlabel(
        "permissive minus shipped, refusals (points)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def power(out: str) -> None:
    cases, runs = _inputs()
    outcomes = compare.question_outcomes(cases, runs)
    names, questions, rates = compare._rates(outcomes)
    differ = sum(x != y for x, y in zip(rates[names[0]], rates[names[1]]))
    noise = differ / len(questions)
    grid = [20, 30, 45, 55, 70, 90, 120, 160, 220, 300, 400, 550, 750, 1000]
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=200)
    for gap, color in ((0.05, GREY), (0.10, BLUE), (0.15, INK)):
        ys = [compare.power_sign_test(n, gap, noise + gap) for n in grid]
        ax.plot(
            grid,
            [100 * y for y in ys],
            color=color,
            lw=2,
            label=f"a real gap of {100 * gap:.0f} points",
        )
    ax.axhline(80, color=GREY, lw=0.8, ls="--")
    ax.axvline(55, color=RED, lw=1)
    ax.text(51, 88, "the 55 questions\nI have", fontsize=8.5, color=RED, ha="right")
    ax.legend(loc="lower right", fontsize=8.5, frameon=False)
    ax.set_xscale("log")
    ax.set_xlim(20, 1500)
    ax.set_ylim(0, 102)
    ax.set_xticks([20, 55, 100, 200, 500, 1000])
    ax.set_xticklabels(["20", "55", "100", "200", "500", "1,000"])
    ax.set_xlabel("questions", fontsize=9, color=INK)
    ax.set_ylabel("chance the test sees a real gap (%)", fontsize=9, color=INK)
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"intervals": intervals, "power": power}[sys.argv[1]](sys.argv[2])
