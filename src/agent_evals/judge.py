"""An LLM judge for one question: is this answer supported by what it was given? (chapter 15)

The judge reads the package information, the question and the answer, lists the claims
the answer makes, and says for each whether the information states it. Its verdict is
`unsupported` if any claim is. It is a model, so it has an error rate of its own, and
this module keeps everything needed to measure that: the prompt, its hash, the raw
reply, the token counts and the time.
"""

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from reliable_agents_labs.json_parsing import parse_json_object

PROMPT_V1 = (
    "You check whether an answer is supported by the package information it was given. "
    "You will see the package information, a question and an answer. List every claim "
    "the answer makes about a package or about the world. For each claim, say whether "
    "the package information states it, or states it in other words. A claim is "
    "unsupported if the package information does not state it, even if it is true, or "
    "if it contradicts the package information. Respond with a single JSON object, no "
    'markdown fences: {"claims": [{"claim": "...", "supported": true or false}], '
    '"verdict": "supported" or "unsupported"}. The verdict is "unsupported" if any '
    "claim is unsupported. If the answer only says that the information is not enough, "
    'it makes no claims, and the verdict is "supported".'
)
VERSIONS = {"v1": PROMPT_V1}


def prompt_hash(version: str) -> str:
    return hashlib.sha256(VERSIONS[version].encode("utf8")).hexdigest()


def load_items(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [json.loads(x) for x in lines if x.strip()]


def user_message(item: dict[str, Any]) -> str:
    context = "\n".join(f"- {c['name']}: {c['summary']}" for c in item["context"])
    return (
        f"Package information:\n{context}\n\nQuestion: {item['question']}\n\n"
        f"Answer: {item['answer']}"
    )


async def judge_item(client, system: str, item: dict[str, Any]) -> dict[str, Any]:
    """One verdict. A reply that is not the JSON asked for is an error, not a guess."""
    started = time.perf_counter()
    row: dict[str, Any] = {"item_id": item["item_id"], "verdict": None, "claims": []}
    try:
        result = await client.generate(system=system, user=user_message(item))
        row["input_tokens"] = result.input_tokens
        row["output_tokens"] = result.output_tokens
        payload = parse_json_object(result.text)
        verdict = payload["verdict"]
        if verdict not in ("supported", "unsupported"):
            raise ValueError(f"unknown verdict {verdict!r}")
        row["verdict"] = verdict
        row["claims"] = payload.get("claims", [])
    except Exception as exc:  # noqa: BLE001 - a bad reply is a result
        row["error"] = f"{type(exc).__name__}: {exc}"
    row["seconds"] = round(time.perf_counter() - started, 3)
    return row


async def judge_all(
    items: list[dict[str, Any]],
    client,
    version: str,
    passes: int = 2,
    concurrency: int = 6,
) -> list[dict[str, Any]]:
    """Every item, `passes` times, at most `concurrency` calls at once."""
    gate = asyncio.Semaphore(concurrency)
    system = VERSIONS[version]

    async def one(item: dict[str, Any], number: int) -> dict[str, Any]:
        async with gate:
            row = await judge_item(client, system, item)
        row["pass"] = number
        return row

    tasks = [one(item, n) for n in range(1, passes + 1) for item in items]
    return list(await asyncio.gather(*tasks))


class ScriptedJudgeClient:
    """A judge that flags an answer when it contains one of `flags`, or replies with
    text that is not JSON when `broken`. It lets the plumbing be tested with no key."""

    def __init__(self, flags: tuple[str, ...] = (), broken: bool = False) -> None:
        self.flags, self.broken = flags, broken

    async def generate(self, *, system, user, tools=None, history=None):
        from reliable_agents_labs.models import ModelResult

        if self.broken:
            text = "I think it is fine."
        else:
            bad = any(f in user.split("Answer:", 1)[1] for f in self.flags)
            payload = {
                "claims": [{"claim": "x", "supported": not bad}],
                "verdict": "unsupported" if bad else "supported",
            }
            text = json.dumps(payload)
        return ModelResult(
            text=text,
            input_tokens=100,
            output_tokens=20,
            model_id="scripted",
            provider="scripted",
            tool_calls=[],
        )
