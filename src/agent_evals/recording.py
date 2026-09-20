"""Record every tool call a model asks for, not only the ones that changed something.

The ledger of side effects (chapter 6) is written by the tools, after they run, so a
call that was refused, that named a tool the specialist was never given, or that only
read something leaves no trace there. This wrapper sits between the product and the
model and writes down each call the model asks for: which tool, with what arguments,
whether that tool was on offer, and what came back.
"""

from typing import Any


class RecordingClient:
    """Wraps a model client. It changes nothing about what the model is asked."""

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.calls: list[dict[str, Any]] = []
        self._round = 0
        # Every call to the model, whether or not it asked for a tool.
        self.model_calls = 0
        # What the provider reported for those calls (chapter 22).
        self.input_tokens = 0
        self.output_tokens = 0
        self.thinking_tokens = 0
        # One list of messages per tool loop, kept so results can be read at the end.
        self._histories: dict[int, list[dict]] = {}
        self._batches: dict[int, list[list[dict[str, Any]]]] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    async def generate(self, *, system, user, tools=None, history=None):
        if history is not None:
            self._histories.setdefault(id(history), history)
        result = await self._inner.generate(
            system=system, user=user, tools=tools, history=history
        )
        self._round += 1
        self.model_calls += 1
        self.input_tokens += getattr(result, "input_tokens", 0) or 0
        self.output_tokens += getattr(result, "output_tokens", 0) or 0
        self.thinking_tokens += getattr(result, "thinking_tokens", 0) or 0
        offered = {t["function"]["name"] for t in tools or []}
        batch = [
            {
                "round": self._round,
                "name": c.name,
                "arguments": dict(c.arguments),
                "offered": c.name in offered,
                "result": None,
            }
            for c in result.tool_calls
        ]
        self.calls += batch
        if batch and history is not None:
            self._batches.setdefault(id(history), []).append(batch)
        return result

    def usage(self) -> dict[str, int]:
        """Model calls and tokens, summed over a run. With a metered client, output
        tokens are everything the provider bills, and thinking_tokens is the part of
        them the model did not show. With any other client thinking_tokens is 0 and
        output tokens are whatever the client reported, which may leave thinking out."""
        return {
            "model_calls": self.model_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "thinking_tokens": self.thinking_tokens,
        }

    def finish(self) -> list[dict[str, Any]]:
        """Attach each call's result and return the calls in the order they were asked."""
        for key, batches in self._batches.items():
            messages = self._histories[key]
            # Each assistant message that asked for tools is followed by one tool
            # message per call, in the same order.
            answered = iter([m for m in messages if m.get("role") == "tool"])
            for batch in batches:
                for call in batch:
                    reply = next(answered, None)
                    if reply is not None:
                        call["result"] = reply.get("content")
        return self.calls
