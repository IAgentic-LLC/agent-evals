"""Plot the retrieval results (chapter 12, figures 12.2 and 12.3).

Reads the recorded embeddings, so the figures cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_retrieval.py METHODS.png SCORES.png
"""

import asyncio
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import retrieval as R
from agent_evals import retrieval_report as RR
from agent_evals.runner import load_cases
from agent_evals.stats import wilson_interval

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs" / "pkg-retrieval-2"
INK = "#1F2937"
BLUE, GREY, RED, AMBER = "#2563EB", "#9CA3AF", "#E03131", "#F08C00"


def load_study() -> RR.Study:
    study = RR.Study(
        R.load_corpus(ROOT / "datasets" / "pkg_corpus_v1.jsonl"),
        load_cases(ROOT / "datasets" / "pkg_queries_v2.jsonl"),
        R.load_vectors(RUN / "embeddings-shipped.npz"),
        R.load_vectors(RUN / "embeddings-typed.npz"),
    )
    return asyncio.run(study.run())


def plot_methods(study: RR.Study, out: str) -> None:
    n = len(study.answerable)
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=200)
    colors = {"random (seed 0)": GREY, "alphabetical": GREY}
    for row, method in enumerate(reversed(RR.METHODS)):
        k = sum(study.hits(method, 3))
        low, high = wilson_interval(k, n)
        color = colors.get(method, BLUE)
        ax.plot(
            [100 * low, 100 * high],
            [row, row],
            color=color,
            lw=7,
            solid_capstyle="butt",
        )
        ax.plot([100 * k / n], [row], "o", color=INK, ms=6, zorder=3)
        ax.text(
            100 * high + 1.5, row, f"{k} of {n}", va="center", fontsize=9, color=INK
        )
    ax.set_yticks(range(len(RR.METHODS)))
    ax.set_yticklabels(list(reversed(RR.METHODS)), fontsize=9.5, color=INK)
    ax.set_xlim(0, 118)
    ax.set_xticks(range(0, 101, 20))
    ax.set_xlabel(
        "Questions with a hit in the top 3 (%, 95% interval)",
        fontsize=9,
        color=INK,
    )
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def plot_scores(study: RR.Study, out: str) -> None:
    top = study.top1["shipped"]
    groups = [
        ("task", [top[q] for q in study.answerable if study.kind[q] == "task"], BLUE),
        ("named", [top[q] for q in study.answerable if study.kind[q] == "named"], BLUE),
        ("hard", [top[q] for q in study.answerable if study.kind[q] == "hard"], AMBER),
        ("no answer", [top[q] for q in study.queries if not study.relevant[q]], RED),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 2.7), dpi=200)
    for row, (label, values, color) in enumerate(reversed(groups)):
        jitter = [row + (i % 5 - 2) * 0.06 for i in range(len(values))]
        ax.scatter(values, jitter, s=22, color=color, alpha=0.85, edgecolors="none")
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels([g[0] for g in reversed(groups)], fontsize=9.5, color=INK)
    ax.set_xlabel(
        "Cosine similarity of the best match, dense as shipped", fontsize=9, color=INK
    )
    ax.tick_params(axis="x", labelsize=9.5, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    study = load_study()
    plot_methods(study, sys.argv[1])
    plot_scores(study, sys.argv[2])
