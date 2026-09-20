"""Test a grader against hand labels: a grader is software, and it can be wrong."""

import json
from dataclasses import dataclass, field
from pathlib import Path

from agent_evals.answer_graders import Grader
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval


@dataclass
class Miss:
    kind: str  # "false positive" or "false negative"
    run: str
    case_id: str
    trial: int
    contested: bool
    answer: str


@dataclass
class Result:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0
    misses: list[Miss] = field(default_factory=list)

    @property
    def total(self) -> int:
        return self.tp + self.fp + self.fn + self.tn


def load_labels(path: str | Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def check(
    grader: Grader,
    labels: list[dict],
    runs_dir: str | Path,
    cases: dict[str, EvalCase],
) -> Result:
    traces: dict[str, dict[tuple[str, int], Trace]] = {}
    result = Result()
    for row in labels:
        run = row["run"]
        if run not in traces:
            found = read_traces(Path(runs_dir) / run / "traces.jsonl")
            traces[run] = {(t.case_id, t.trial): t for t in found}
        trace = traces[run][(row["case_id"], row["trial"])]
        flagged = grader(cases[row["case_id"]], trace)
        truth = row["label"]
        if flagged and truth:
            result.tp += 1
        elif flagged and not truth:
            result.fp += 1
        elif truth:
            result.fn += 1
        else:
            result.tn += 1
        if flagged != truth:
            kind = "false positive" if flagged else "false negative"
            result.misses.append(
                Miss(
                    kind,
                    run,
                    row["case_id"],
                    row["trial"],
                    row["contested"],
                    " ".join(trace.answer.split()),
                )
            )
    return result


def load_cases_by_id(*paths: str | Path) -> dict[str, EvalCase]:
    return {c.case_id: c for p in paths for c in load_cases(p)}


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "n/a"
    low, high = wilson_interval(k, n)
    return f"{k}/{n} ({100 * k / n:.1f}%)  {100 * low:.0f}-{100 * high:.0f}%"


def render(name: str, result: Result, snippet: int = 70) -> str:
    positives = result.tp + result.fn
    lines = [
        f"Grader {name}: {result.total} labeled answers ({positives} yes)",
        "",
        "                labeled yes  labeled no",
        f"  flagged yes   {result.tp:>11}  {result.fp:>10}",
        f"  flagged no    {result.fn:>11}  {result.tn:>10}",
        "",
        f"Precision  {_rate(result.tp, result.tp + result.fp)}",
        f"Recall     {_rate(result.tp, positives)}",
    ]
    if result.misses:
        lines += ["", "Errors:"]
        for m in result.misses:
            mark = " (contested)" if m.contested else ""
            lines.append(f"  {m.kind}{mark}: {m.run} {m.case_id} t{m.trial}")
            lines.append(f"    {m.answer[:snippet]}")
    return "\n".join(lines) + "\n"
