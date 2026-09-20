"""Live check of the token meter against the provider's own counts (chapter 22).

For each model it makes one real call through the metered client and prints what the
provider reported for that call next to what the meter made of it: `written` is the
count the product's own client uses, `total` is the provider's total, `billed` and
`thinking` are the meter's figures, and `agrees` says the meter kept the provider's
total. Then it makes a second call to the provider's native interface, which reports
thinking tokens in a field of their own, and prints that count as `native`. The two
calls are separate, so their thinking differs in length; the point is that the
native interface names thinking tokens too, and that its parts add up to its total.

This spends a fraction of a cent. The numbers change from call to call, because how
long a model thinks changes.

Usage: uv run python scripts/check_meter_live.py [ENV_FILE]
"""

import asyncio
import os
import sys

import httpx
from dotenv import load_dotenv

from agent_evals.metering import MeteredGeminiClient

MODELS = ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash")
SYSTEM = "You are a billing support agent."
USER = (
    "A customer was charged twice for invoice INV-2041, $84.50 each. "
    "Which charge should be refunded, and why?"
)
NATIVE = "https://generativelanguage.googleapis.com/v1beta/models/{}:generateContent"


async def native_thinking(model: str) -> str:
    """Thinking tokens the native interface reports for the same prompt, or "-"."""
    body = {
        "systemInstruction": {"parts": [{"text": SYSTEM}]},
        "contents": [{"parts": [{"text": USER}]}],
    }
    headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
    async with httpx.AsyncClient(timeout=120) as http:
        response = await http.post(NATIVE.format(model), json=body, headers=headers)
    if response.status_code != 200:
        return "-"
    usage = response.json()["usageMetadata"]
    parts = (
        usage.get("promptTokenCount", 0)
        + usage.get("candidatesTokenCount", 0)
        + usage.get("thoughtsTokenCount", 0)
    )
    assert parts == usage["totalTokenCount"], usage
    return str(usage.get("thoughtsTokenCount", 0))


async def one(model: str) -> str:
    client = MeteredGeminiClient(model_id=model)
    seen = []
    real = client._client.chat.completions.create

    async def spy(**kwargs):
        response = await real(**kwargs)
        seen.append(response.usage)
        return response

    client._client.chat.completions.create = spy
    result = await client.generate(system=SYSTEM, user=USER)
    u = seen[0]
    agrees = result.input_tokens + result.output_tokens == u.total_tokens
    return (
        f"{model.removeprefix('gemini-'):<16}{u.prompt_tokens:>6}"
        f"{u.completion_tokens:>9}{u.total_tokens:>7}"
        f"{result.output_tokens:>8}{result.thinking_tokens:>10}"
        f"{'yes' if agrees else 'NO':>8}{await native_thinking(model):>8}"
    )


async def main() -> None:
    load_dotenv(sys.argv[1] if len(sys.argv) > 1 else "../pkgintel-app/.env")
    print(
        f"{'model':<16}{'input':>6}{'written':>9}{'total':>7}"
        f"{'billed':>8}{'thinking':>10}{'agrees':>8}{'native':>8}"
    )
    for model in MODELS:
        print(await one(model))


asyncio.run(main())
