"""Write the items a judge is tested on (datasets/judge_items_v1.jsonl), chapter 15.

Two groups, built from the recorded answers of chapter 13 and my readings of them:

- `real`: the 92 answers that cite something (my readings: 85 supported, 6 borderline,
  1 stretch). The label is my reading, made before any judge existed.
- `planted`: 45 copies of answers I read as supported, each with one sentence added that
  the retrieved summaries do not state. Three kinds, 15 each: `praise` (an evaluation),
  `fact` (a specific fact about the package) and `capability` (something it can do).
  The truth is known by construction, and the sentences are true or not, which does not
  matter: the rubric asks whether the information states it.

The 45 clean originals are the `real` items with `clean_of_planted` set, so a judge is
scored on 45 faulty and 45 clean answers. The real items alternate between `dev` and
`test`, in order, and a faulty copy is in the same half as its original, with the three
kinds cycling within each half. The random
choices use seed 0.
"""

import json
import random
from pathlib import Path

from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "datasets" / "judge_items_v1.jsonl"
RUNS = ("pkg-answers-1", "pkg-answers-2")
SENTENCES = {
    "praise": [
        "It is the most widely used choice among professional teams.",
        "Most developers consider it the best option available.",
        "It is generally regarded as the industry standard.",
        "It is very easy to learn and popular with beginners.",
        "It has an excellent reputation for reliability.",
    ],
    "fact": [
        "It was originally created by a small team of volunteers.",
        "It is maintained by a large open-source foundation.",
        "It was created by a research group at a university.",
        "Its source code is hosted on GitHub.",
        "It has been part of the Python ecosystem for many years.",
    ],
    "capability": [
        "It can also convert files between several formats.",
        "It also includes a command line interface.",
        "It works without any extra dependencies.",
        "It supports running on many machines at once.",
        "It can export its results to a spreadsheet.",
    ],
}


def line(item: dict) -> str:
    return json.dumps(item, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets" / "pkg_answers_v1.jsonl")
    }
    readings = {}
    for row in (
        (ROOT / "datasets" / "pkg_answers_v1.readings.jsonl")
        .read_text(encoding="utf8")
        .splitlines()
    ):
        r = json.loads(row)
        readings[(r["run"], r["case_id"])] = r["reading"]
    real = []
    for run in RUNS:
        for t in read_traces(ROOT / "runs" / run / "traces.jsonl"):
            reading = readings[(run, t.case_id)]
            if reading == "refusal":
                continue
            real.append(
                {
                    "item_id": f"{run}:{t.case_id}",
                    "group": "real",
                    "kind": "real",
                    "label": "supported" if reading == "supported" else "unsupported",
                    "reading": reading,
                    "question": cases[t.case_id].input["query"],
                    "context": [
                        {"name": r["name"], "summary": r["summary"]}
                        for r in t.retrieved
                    ],
                    "answer": t.answer,
                    "clean_of_planted": False,
                }
            )
    for number, item in enumerate(real):
        item["split"] = "dev" if number % 2 == 0 else "test"
    rng = random.Random(0)
    pool = [
        i
        for i in real
        if i["reading"] == "supported"
        and not i["item_id"].endswith(tuple(f":CF-{n:03d}" for n in range(1, 6)))
    ]
    chosen = rng.sample(pool, 45)
    planted = []
    kinds = ["praise", "fact", "capability"]
    # The kinds cycle within each half, so each kind is spread across dev and test.
    for split in ("dev", "test"):
        for number, source in enumerate(c for c in chosen if c["split"] == split):
            kind = kinds[number % 3]
            source["clean_of_planted"] = True
            sentence = rng.choice(SENTENCES[kind])
            planted.append(
                {
                    "item_id": f"planted:{kind}:{source['item_id']}",
                    "group": "planted",
                    "kind": kind,
                    "label": "unsupported",
                    "reading": "planted",
                    "question": source["question"],
                    "context": source["context"],
                    "answer": f"{source['answer']} {sentence}",
                    "clean_of_planted": False,
                    "added": sentence,
                    "source_id": source["item_id"],
                    "split": split,
                }
            )
    OUT.write_text(
        "\n".join(line(i) for i in [*real, *planted]) + "\n",
        encoding="utf8",
        newline="\n",
    )
    print(f"wrote {len(real)} real and {len(planted)} planted items to {OUT.name}")


if __name__ == "__main__":
    main()
