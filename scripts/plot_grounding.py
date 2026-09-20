"""Plot the grounding results (chapter 13, figures 13.2 and 13.3).

Reads the recorded runs and my hand readings, and runs the scripted check, so the
figures cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_grounding.py READINGS.png MATRIX.png
"""

import asyncio
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import grounding
from agent_evals.adapters.pkgintel import PkgAnswerAdapter, ScriptedRagClient
from agent_evals.grounding_check import FAULTS
from agent_evals.retrieval import load_corpus
from agent_evals.runner import load_cases, run_cases

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
BLUE, GREY, RED, AMBER, GREEN = "#2563EB", "#9CA3AF", "#E03131", "#F08C00", "#2F9E44"
DATASET = ROOT / "datasets" / "pkg_answers_v1.jsonl"
NAMES = [r["name"] for r in load_corpus(ROOT / "datasets" / "pkg_corpus_v1.jsonl")]


def plot_readings(out: str) -> None:
    cases = {c.case_id: c for c in load_cases(DATASET)}
    readings = grounding.load_readings(
        ROOT / "datasets" / "pkg_answers_v1.readings.jsonl"
    )
    kinds = list(grounding.KINDS)
    colors = {"supported": BLUE, "refusal": GREY, "stretch": RED}
    fig, ax = plt.subplots(figsize=(5.5, 2.9), dpi=200)
    for row, kind in enumerate(reversed(kinds)):
        left = 0
        for reading in ("supported", "refusal", "stretch"):
            n = sum(
                r["reading"] == reading and cases[r["case_id"]].slices["kind"] == kind
                for r in readings
            )
            if n:
                ax.barh(row, n, left=left, color=colors[reading], height=0.62)
                if n >= 6:
                    ax.text(
                        left + n / 2,
                        row,
                        str(n),
                        ha="center",
                        va="center",
                        fontsize=8.5,
                        color="white",
                    )
                left += n
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(list(reversed(kinds)), fontsize=9.5, color=INK)
    ax.set_xlabel("Answers, two runs together (154)", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    ax.legend(
        handles,
        [f"read as {k}" for k in colors],
        fontsize=8,
        frameon=False,
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


async def _matrix() -> dict[str, dict[str, int]]:
    cases = load_cases(DATASET)
    by_id = {c.case_id: c for c in cases}
    table: dict[str, dict[str, int]] = {}
    for mode in ("faithful", *FAULTS, "adds_claim"):
        adapter = PkgAnswerAdapter(client=ScriptedRagClient(mode, NAMES))
        traces = await run_cases(cases, adapter, concurrency=1)
        row = {flag: 0 for flag in grounding.FLAGS}
        for t in traces:
            found = grounding.check(by_id[t.case_id], t, NAMES)
            for flag in grounding.FLAGS:
                row[flag] += bool(found[flag])
        table[mode] = row
    return table


def plot_matrix(out: str) -> None:
    table = asyncio.run(_matrix())
    flags = list(grounding.FLAGS)
    modes = list(table)
    labels = {
        "faithful": "faithful",
        "cites_unretrieved": "cites a package not retrieved",
        "names_outside": "names a package not retrieved",
        "states_version": "states a version number",
        "cites_nothing": "cites nothing",
        "ignores_context": "ignores the edited context",
        "invalid_json": "returns text, not JSON",
        "adds_claim": "adds an unsupported claim",
    }
    fig, ax = plt.subplots(figsize=(5.5, 3.3), dpi=200)
    for i, mode in enumerate(modes):
        for j, flag in enumerate(flags):
            n = table[mode][flag]
            aimed = FAULTS.get(mode) == flag
            fill = BLUE if aimed else ("white" if n == 0 else AMBER)
            ax.add_patch(
                plt.Rectangle(
                    (j, -i), 0.94, 0.86, facecolor=fill, edgecolor=GREY, lw=0.8
                )
            )
            ax.text(
                j + 0.47,
                -i + 0.43,
                str(n),
                ha="center",
                va="center",
                fontsize=8.5,
                color="white" if aimed else INK,
            )
    ax.set_xlim(-0.05, len(flags))
    ax.set_ylim(-len(modes) + 0.9, 1.35)
    ax.set_xticks([j + 0.47 for j in range(len(flags))])
    ax.set_xticklabels([f.replace("_", "\n") for f in flags], fontsize=8, color=INK)
    ax.xaxis.tick_top()
    ax.set_yticks([-i + 0.43 for i in range(len(modes))])
    ax.set_yticklabels([labels[m] for m in modes], fontsize=8.5, color=INK)
    ax.tick_params(length=0)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    plot_readings(sys.argv[1])
    plot_matrix(sys.argv[2])
