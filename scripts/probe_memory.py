"""What the model says about the latest versions when nothing constrains it (chapter 13).

The product's prompt tells the model to answer only from the retrieved text. This
probe removes that: it asks the same ten freshness questions with no context and no
such instruction, twice, and records what comes back. It is not a test of the product.
It shows what the version check protects against.

Usage: uv run python scripts/probe_memory.py OUT_DIR --env-file ENV
       uv run python scripts/probe_memory.py OUT_DIR --show
"""

import argparse
import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from agent_evals.grounding import VERSION
from agent_evals.manifest import harness_state, now
from agent_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets" / "pkg_answers_v1.jsonl"
SYSTEM = "Answer in one short sentence."


async def record(out: Path) -> None:
    from reliable_agents_labs.models import build_model_client

    client = build_model_client("answer_model")
    cases = [c for c in load_cases(DATASET) if c.slices["kind"] == "fresh"]
    harness, started = harness_state(), now()

    async def one(case, trial):
        result = await client.generate(system=SYSTEM, user=case.input["query"])
        return {
            "case_id": case.case_id,
            "trial": trial,
            "package": case.expected["relevant"][0],
            "answer": result.text,
        }

    rows = await asyncio.gather(*(one(c, t) for c in cases for t in (1, 2)))
    out.mkdir(parents=True, exist_ok=True)
    (out / "answers.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8"
    )
    manifest = {
        "note": "no context, no grounding instruction, system prompt: " + SYSTEM,
        "dataset": DATASET.name,
        "started_at": started.isoformat(timespec="seconds"),
        "harness": harness,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def show(out: Path) -> None:
    cases = {c.case_id: c for c in load_cases(DATASET)}
    rows = [
        json.loads(x)
        for x in (out / "answers.jsonl").read_text(encoding="utf8").splitlines()
    ]
    print(f"{'case':<8}{'package':<14}{'PyPI, 20 Sep':<14}from memory, two tries")
    for case_id in sorted({r["case_id"] for r in rows}):
        said = []
        for r in sorted(
            (r for r in rows if r["case_id"] == case_id), key=lambda r: r["trial"]
        ):
            versions = VERSION.findall(r["answer"])
            said.append(", ".join(versions) if versions else "none")
        case = cases[case_id]
        print(
            f"{case_id:<8}{case.expected['relevant'][0]:<14}"
            f"{case.expected['version']:<14}{' | '.join(said)}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--env-file")
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.show:
        show(args.out)
    else:
        if args.env_file:
            load_dotenv(args.env_file)
        asyncio.run(record(args.out))
        show(args.out)
