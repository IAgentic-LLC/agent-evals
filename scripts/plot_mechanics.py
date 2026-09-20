"""Plot which mechanics checks each workflow variant fails (chapter 11, figure 11.2).

Runs the scripted checks, so the figure is the result and not a drawing of it.

Usage: uv run --with matplotlib python scripts/plot_mechanics.py OUTPUT.png
"""

import asyncio
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import mechanics

INK = "#1F2937"
PASS, FAIL = "#D3F9D8", "#E03131"
LABELS = {
    "the real workflow": "the real\nworkflow",
    "acts before approval": "acts\nbefore\napproval",
    "pauses inside the model node": "pauses\ninside the\nmodel node",
    "logs the action twice": "logs the\naction\ntwice",
    "forgets on restart": "forgets\non\nrestart",
    "one thread for everyone": "one\nthread for\neveryone",
}


def main(out: str) -> None:
    variants = list(mechanics.VARIANTS.values())
    results = [asyncio.run(mechanics.run_mechanics(v)) for v in variants]
    checks = list(mechanics.CHECKS)
    fig, ax = plt.subplots(figsize=(5.5, 3.9), dpi=200)
    for col, found in enumerate(results):
        for row, check in enumerate(checks):
            failed = bool(found[check])
            ax.add_patch(
                plt.Rectangle(
                    (col, row), 0.92, 0.88, color=FAIL if failed else PASS, lw=0
                )
            )
            ax.text(
                col + 0.46,
                row + 0.44,
                "fails" if failed else "passes",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if failed else INK,
            )
    ax.set_xlim(0, len(variants))
    ax.set_ylim(len(checks), 0)
    ax.set_yticks([r + 0.44 for r in range(len(checks))])
    ax.set_yticklabels([c.replace("_", " ") for c in checks], fontsize=7.5, color=INK)
    ax.set_xticks([c + 0.46 for c in range(len(variants))])
    ax.set_xticklabels([LABELS[v.name] for v in variants], fontsize=7, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
