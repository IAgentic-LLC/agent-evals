"""Plot the data figures of chapter 26 from the recorded runs.

`passes`: the rank of each entry on a board made from one pass of each, for each of the
three passes. Lines that cross are entries whose order changes when the run is repeated.
`frontier`: the met rate against the cost of a success, with the interval on each entry,
the entries that broke an invariant shown as crosses, and the cheapest way to each level.

Usage: uv run --with matplotlib python scripts/plot_leaderboard.py passes OUT.png
       uv run --with matplotlib python scripts/plot_leaderboard.py frontier OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import leaderboard

ROOT = Path(__file__).resolve().parents[1]
INK, GREY, BLUE, RED, AMBER = "#1F2937", "#9CA3AF", "#2563EB", "#DC2626", "#F59E0B"


def _style(ax) -> None:
    ax.tick_params(labelsize=9, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def _rows():
    board = leaderboard.load_board(ROOT / "leaderboards/triage_heldout_v1.yaml")
    rows, _, _ = leaderboard.build(ROOT, board)
    return board, rows


def passes(out: str) -> None:
    board, rows = _rows()
    picked = [r for r in rows if r.label in board.first_look]
    ranks = [leaderboard._rank([r.passes[k] for r in picked]) for k in range(3)]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    colors = [BLUE, GREY, AMBER, RED, "#059669", "#7C3AED"]
    for i, r in enumerate(picked):
        ys = [ranks[k][i] for k in range(3)]
        ax.plot([1, 2, 3], ys, "o-", color=colors[i % len(colors)], lw=2, ms=5)
        ax.text(3.06, ys[2], r.label, fontsize=8, va="center", color=INK)
        ax.text(0.94, ys[0], r.label, fontsize=8, va="center", ha="right", color=INK)
    ax.set_xlim(-0.7, 4.2)
    ax.set_ylim(6.4, 0.6)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["pass 1", "pass 2", "pass 3"], fontsize=9)
    ax.set_yticks(range(1, 7))
    ax.set_ylabel("rank on a board of one pass each", fontsize=9, color=INK)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


def frontier(out: str) -> None:
    _, rows = _rows()
    ranked = [r for r in rows if not r.blocked and r.cost_per_success is not None]
    blocked = [r for r in rows if r.blocked and r.cost_per_success is not None]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=200)
    offsets = {
        "3.5-lite + note": (-40, 40),
        "3.5-lite + note + guard": (-40, -66),
    }
    for r in ranked:
        x = 1000 * r.cost_per_success
        top = 100 * r.interval[1]
        ax.plot([x, x], [100 * r.interval[0], top], color=GREY, lw=2)
        ax.plot([x], [100 * r.met], "o", color=BLUE, ms=6)
        if r.label in offsets:
            ax.annotate(
                r.label,
                (x, 100 * r.met),
                textcoords="offset points",
                xytext=offsets[r.label],
                fontsize=7.5,
                color=INK,
                arrowprops={"arrowstyle": "-", "color": GREY, "lw": 0.8},
            )
        else:
            ax.text(x, top + 1.2, r.label, fontsize=7.5, color=INK, ha="center")
    blocked_offsets = {
        "2.5-flash": (10, 6),
        "3.5-lite + guard": (12, 0),
        "3.5-lite": (12, -22),
    }
    for r in blocked:
        x = 1000 * r.cost_per_success
        ax.plot([x], [100 * r.met], "x", color=RED, ms=7, mew=2)
        ax.annotate(
            r.label + " (blocked)",
            (x, 100 * r.met),
            textcoords="offset points",
            xytext=blocked_offsets.get(r.label, (10, 4)),
            fontsize=7.5,
            color=RED,
        )
    ax.set_xlim(-0.5, 6.8)
    ax.set_xlabel("dollars per success (thousandths)", fontsize=9, color=INK)
    ax.set_ylabel("runs meeting the required actions (%)", fontsize=9, color=INK)
    ax.set_ylim(40, 100)
    _style(ax)
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    {"passes": passes, "frontier": frontier}[sys.argv[1]](sys.argv[2])
