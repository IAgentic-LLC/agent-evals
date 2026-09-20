"""Live check of the token meter against the provider's own count (chapter 22).

For each model it makes one real call and prints what the provider reported, and what
the metered client made of the same reply. The two must agree that input plus output
equals the provider's total. This spends a fraction of a cent. The numbers change from
call to call, because how long a model thinks changes; the agreement does not.

Usage: uv run python scripts/check_meter_live.py [ENV_FILE]
"""

import asyncio
import sys

from dotenv import load_dotenv

from agent_evals.metering import MeteredGeminiClient

MODELS = ("gemini-3.6-flash", "gemini-3.5-flash-lite", "gemini-2.5-flash")
SYSTEM = "You are a billing support agent."
USER = (
    "A customer was charged twice for invoice INV-2041, $84.50 each. "
    "Which charge should be refunded, and why?"
)


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
        f"{model.removeprefix('gemini-'):<17}{u.prompt_tokens:>6}"
        f"{u.completion_tokens:>9}{u.total_tokens:>7}"
        f"{result.output_tokens:>8}{result.thinking_tokens:>10}"
        f"{'yes' if agrees else 'NO':>8}"
    )


async def main() -> None:
    load_dotenv(sys.argv[1] if len(sys.argv) > 1 else "../pkgintel-app/.env")
    print(
        f"{'model':<17}{'input':>6}{'written':>9}{'total':>7}"
        f"{'billed':>8}{'thinking':>10}{'agrees':>8}"
    )
    for model in MODELS:
        print(await one(model))


asyncio.run(main())
