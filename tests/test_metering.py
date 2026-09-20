"""Chapter 22: the meter counts what the provider bills, thinking tokens included.

The numbers in the fake usage record are the ones the live endpoint returned on
2026-09-21 for one call to gemini-3.6-flash: 47 in, 470 written, 1402 in total.
"""

from types import SimpleNamespace

from agent_evals import metering
from agent_evals.recording import RecordingClient


def _usage(prompt, completion, total):
    return SimpleNamespace(
        prompt_tokens=prompt, completion_tokens=completion, total_tokens=total
    )


def test_output_is_the_total_less_the_input_and_thinking_is_what_was_not_written():
    assert metering.billed_tokens(_usage(47, 470, 1402)) == (47, 1355, 885)


def test_a_reply_with_no_thinking_has_the_written_tokens_as_its_output():
    assert metering.billed_tokens(_usage(100, 60, 160)) == (100, 60, 0)


def test_a_total_too_small_to_hold_the_written_tokens_never_lowers_the_output():
    assert metering.billed_tokens(_usage(100, 60, 120)) == (100, 60, 0)


def test_a_missing_usage_record_counts_as_nothing_and_not_an_error():
    assert metering.billed_tokens(None) == (0, 0, 0)


class _Completions:
    def __init__(self, usage):
        self._usage = usage

    async def create(self, **kwargs):
        message = SimpleNamespace(content="Done.", tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message)], usage=self._usage
        )


def _client(usage):
    client = metering.MeteredGeminiClient.__new__(metering.MeteredGeminiClient)
    client._model_id = "gemini-3.6-flash"
    client._client = SimpleNamespace(
        chat=SimpleNamespace(completions=_Completions(usage))
    )
    return client


async def test_the_client_returns_the_billed_output_and_the_thinking_part_of_it():
    result = await _client(_usage(47, 470, 1402)).generate(system="s", user="u")
    assert result.text == "Done."
    assert (result.input_tokens, result.output_tokens) == (47, 1355)
    assert result.thinking_tokens == 885


async def test_the_recorder_adds_thinking_tokens_across_the_calls_of_a_run():
    recorder = RecordingClient(_client(_usage(47, 470, 1402)))
    await recorder.generate(system="s", user="u")
    await recorder.generate(system="s", user="u")
    assert recorder.usage() == {
        "model_calls": 2,
        "input_tokens": 94,
        "output_tokens": 2710,
        "thinking_tokens": 1770,
    }
