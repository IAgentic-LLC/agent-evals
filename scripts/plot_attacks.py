"""Plot how many faulty answers each attack got past the judge (chapter 16, figure 16.3).

Reads the recorded verdicts, so the figure cannot drift from the data.

Usage: uv run --with matplotlib python scripts/plot_attacks.py OUT.png
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from agent_evals import judge, judge_report

ROOT = Path(__file__).resolve().parents[1]
INK = "#1F2937"
RED, BLUE = "#E03131", "#2563EB"


def passed(items, base_run, perturbed_items, perturbed_run, kind):
    """Faulty verdicts that were `unsupported` before the attack and `supported` after."""
    base = judge_report._verdicts(judge_report.load_rows([ROOT / "runs" / base_run]))
    moved = judge_report._verdicts(
        judge_report.load_rows([ROOT / "runs" / perturbed_run])
    )
    ids = {i["item_id"]: i for i in items}
    k = n = 0
    for p in perturbed_items:
        if p["perturbation"] != kind or ids[p["base_id"]]["group"] != "planted":
            continue
        for number in (1, 2):
            a, b = base.get((p["base_id"], number)), moved.get((p["item_id"], number))
            if a and b:
                n += 1
                k += a == "unsupported" and b == "supported"
    return k, n


def main(out: str) -> None:
    items = judge.load_items(ROOT / "datasets" / "judge_items_v1.jsonl")
    basic = judge.load_items(ROOT / "datasets" / "judge_items_ch16_v1.jsonl")
    attacks = judge.load_items(ROOT / "datasets" / "judge_items_ch16_attacks_v1.jsonl")
    rows = [
        (
            "fault moved to the start",
            basic,
            "judge-ch16-perturbed",
            "fault_first",
            None,
        ),
        ("note to the judge", basic, "judge-ch16-perturbed", "injection", None),
        (
            "note asking for JSON",
            attacks,
            "judge-ch16-attacks",
            "injection_json",
            "judge-v3-attacks",
        ),
        (
            "claimed approval",
            attacks,
            "judge-ch16-attacks",
            "authority",
            "judge-v3-attacks",
        ),
        (
            "fake package information",
            attacks,
            "judge-ch16-attacks",
            "fake_source",
            "judge-v3-attacks",
        ),
    ]
    fig, ax = plt.subplots(figsize=(5.5, 3.0), dpi=200)
    for i, (label, pitems, run, kind, v3run) in enumerate(reversed(rows)):
        k2, n2 = passed(items, "judge-v2-all", pitems, run, kind)
        ax.barh(i + 0.17, k2, height=0.3, color=RED)
        ax.text(k2 + 1, i + 0.17, f"{k2} of {n2}", va="center", fontsize=8, color=RED)
        if v3run:
            k3, n3 = passed(items, "judge-v3-all", pitems, v3run, kind)
            ax.barh(i - 0.17, k3, height=0.3, color=BLUE)
            ax.text(
                k3 + 1, i - 0.17, f"{k3} of {n3}", va="center", fontsize=8, color=BLUE
            )
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)], fontsize=9, color=INK)
    ax.set_xlim(0, 60)
    ax.set_xlabel("Faulty verdicts that moved to supported", fontsize=9, color=INK)
    ax.tick_params(axis="x", labelsize=9, colors=INK)
    ax.tick_params(axis="y", length=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (RED, BLUE)]
    ax.legend(
        handles,
        ["prompt v2", "prompt v3"],
        fontsize=8,
        frameon=False,
        loc="upper right",
    )
    fig.tight_layout()
    fig.savefig(out, facecolor="white")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
