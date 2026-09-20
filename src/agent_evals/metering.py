"""A model client that counts every token the provider bills, including thinking tokens.

Gemini's OpenAI-compatible endpoint reports `completion_tokens` for the text the model
wrote, and leaves the thinking tokens out. They are billed at the output price, and they
are inside `total_tokens`. The library client the product ships with reads only
`completion_tokens`, so a meter built on it undercounts. This client asks the same
endpoint the same question, and reads the total as well (chapter 22).
"""

import json

from reliable_agents_labs.models import (
    GeminiOpenAICompatibleClient,
    ModelResult,
    ToolCall,
)


class MeteredResult(ModelResult):
    """A model result that says how many of its output tokens were thinking."""

    thinking_tokens: int = 0


def billed_tokens(usage) -> tuple[int, int, int]:
    """(input, output including thinking, thinking) from a provider usage record.

    The record's total is input plus everything the model produced, so output is the
    total less the input. If a provider ever reports a total that is too small, the
    reported completion tokens are kept and nothing is invented."""
    if usage is None:
        return 0, 0, 0
    tokens_in = usage.prompt_tokens or 0
    written = usage.completion_tokens or 0
    total = usage.total_tokens or 0
    output = max(written, total - tokens_in)
    return tokens_in, output, output - written


class MeteredGeminiClient(GeminiOpenAICompatibleClient):
    async def generate(
        self,
        *,
        system: str,
        user: str,
        tools: list[dict] | None = None,
        history: list[dict] | None = None,
    ) -> MeteredResult:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if history:
            messages.extend(history)
        kwargs = {"tools": tools} if tools else {}
        response = await self._client.chat.completions.create(
            model=self._model_id, messages=messages, **kwargs
        )
        choice = response.choices[0]
        tokens_in, tokens_out, thinking = billed_tokens(response.usage)
        return MeteredResult(
            text=choice.message.content or "",
            input_tokens=tokens_in,
            output_tokens=tokens_out,
            thinking_tokens=thinking,
            model_id=self._model_id,
            provider="gemini",
            tool_calls=[
                ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                    raw=tc.model_dump(),
                )
                for tc in (choice.message.tool_calls or [])
            ],
        )
