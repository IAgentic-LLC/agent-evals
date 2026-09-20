"""Choose the answers a second person labels, and build the labeling page (chapter 17).

The sample is 50 items from `judge_items_v1`, drawn so that the hard cases are in it:

- every real answer I read as borderline or stretch (7),
- every real answer I read as supported that a judge (prompt v2 or v3, first pass) called
  unsupported,
- real answers I read as supported, one per question, at random, to fill 42,
- 8 planted faults, at random.

The labeler sees the question, the package information and the answer, in a shuffled
order. They see no label of mine, no judge verdict and no stratum. Seed 1.

Writes `datasets/human_label_sample_v1.jsonl` (item ids and strata, for the analysis)
and `label-tool/index.html` (the page, with the items inside it).
"""

import json
import random
from pathlib import Path

from agent_evals import judge, judge_report

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "datasets" / "human_label_sample_v1.jsonl"
PAGE = ROOT / "label-tool" / "index.html"
TEMPLATE = ROOT / "label-tool" / "template.html"


def first_pass(run: str) -> dict[str, str]:
    rows = judge_report.load_rows([ROOT / "runs" / run])
    return {r["item_id"]: r["verdict"] for r in rows if r["pass"] == 1}


def main() -> None:
    items = judge.load_items(ROOT / "datasets" / "judge_items_v1.jsonl")
    real = [i for i in items if i["group"] == "real"]
    planted = [i for i in items if i["group"] == "planted"]
    v2, v3 = first_pass("judge-v2-all"), first_pass("judge-v3-all")
    rng = random.Random(1)

    chosen: dict[str, str] = {}
    for i in real:
        if i["reading"] != "supported":
            chosen[i["item_id"]] = "author: borderline or stretch"
    for i in real:
        flagged = "unsupported" in (v2[i["item_id"]], v3[i["item_id"]])
        if i["reading"] == "supported" and flagged:
            chosen[i["item_id"]] = "author: supported, a judge disagrees"
    taken = {i["question"] for i in real if i["item_id"] in chosen}
    rest = [
        i for i in real if i["item_id"] not in chosen and i["question"] not in taken
    ]
    rng.shuffle(rest)
    for i in rest:
        if len(chosen) >= 42:
            break
        if i["question"] in taken:
            continue
        taken.add(i["question"])
        chosen[i["item_id"]] = "author: supported"
    for i in rng.sample(planted, 8):
        chosen[i["item_id"]] = "planted fault"

    by_id = {i["item_id"]: i for i in items}
    order = list(chosen)
    rng.shuffle(order)
    SAMPLE.write_text(
        "".join(
            json.dumps({"item_id": k, "stratum": chosen[k]}, ensure_ascii=False) + "\n"
            for k in order
        ),
        encoding="utf8",
        newline="\n",
    )
    shown = [
        {
            "id": k,
            "question": by_id[k]["question"],
            "context": by_id[k]["context"],
            "answer": by_id[k]["answer"],
        }
        for k in order
    ]
    html = TEMPLATE.read_text(encoding="utf8").replace(
        "__ITEMS__",
        json.dumps(shown, ensure_ascii=False).replace("</", "<" + chr(92) + "/"),
    )
    PAGE.write_text(html, encoding="utf8", newline="\n")
    counts: dict[str, int] = {}
    for stratum in chosen.values():
        counts[stratum] = counts.get(stratum, 0) + 1
    print(len(chosen), counts)


if __name__ == "__main__":
    main()
