"""Load datasets, run an adapter over cases, and read and write traces."""

import json
from pathlib import Path
from typing import Protocol

from agent_evals.schema import EvalCase, Trace


class Adapter(Protocol):
    name: str

    async def run(self, case: EvalCase, trial: int) -> Trace: ...


def load_cases(path: str | Path) -> list[EvalCase]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [EvalCase.model_validate_json(line) for line in lines if line.strip()]


def write_traces(path: str | Path, traces: list[Trace]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        "".join(t.model_dump_json() + "\n" for t in traces), encoding="utf8"
    )


def read_traces(path: str | Path) -> list[Trace]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [Trace.model_validate_json(line) for line in lines if line.strip()]


async def run_cases(
    cases: list[EvalCase], adapter: Adapter, trials: int = 1
) -> list[Trace]:
    traces: list[Trace] = []
    for case in cases:
        for trial in range(1, trials + 1):
            traces.append(await adapter.run(case, trial))
    return traces


def dump_json(path: str | Path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf8")
