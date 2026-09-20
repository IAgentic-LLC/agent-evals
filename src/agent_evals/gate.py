"""Release gates. Hard gates are counts that must hold; quality gates are thresholds."""

import textwrap
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agent_evals.schema import Scorecard
from agent_evals.stats import min_cases_for_perfect_run


class Rule(BaseModel):
    id: str
    kind: Literal["hard", "quality"]
    metric: str
    op: Literal["min", "max"]
    value: float


class Policy(BaseModel):
    name: str
    rules: list[Rule]


class RuleResult(BaseModel):
    id: str
    kind: str
    metric: str
    observed: float
    op: str
    value: float
    passed: bool
    note: str = ""


class GateResult(BaseModel):
    policy: str
    passed: bool
    results: list[RuleResult]


def _metrics(sc: Scorecard) -> dict[str, float | None]:
    return {
        "routing_rate": sc.routing_rate,
        "routing_rate_lower": sc.routing_interval.low,
        "invariant_violations": float(sc.invariant_violations),
        "latency_p95_s": sc.latency_p95_s,
    }


def load_policy(path: str | Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf8")))


def evaluate(
    policy: Policy, sc: Scorecard, extra: dict[str, float] | None = None
) -> GateResult:
    metrics = {**_metrics(sc), **(extra or {})}
    results: list[RuleResult] = []
    for rule in policy.rules:
        observed = metrics.get(rule.metric)
        if observed is None:
            raise ValueError(
                f"metric {rule.metric!r} is not available for this run; "
                "does the gate need an option such as --invariants?"
            )
        ok = observed >= rule.value if rule.op == "min" else observed <= rule.value
        note = ""
        if not ok and rule.metric == "routing_rate_lower" and rule.op == "min":
            needed = min_cases_for_perfect_run(rule.value)
            if sc.observations < needed:
                note = (
                    f"with {sc.observations} observations even a perfect run cannot "
                    f"pass this rule; it needs at least {needed}"
                )
        results.append(
            RuleResult(
                id=rule.id,
                kind=rule.kind,
                metric=rule.metric,
                observed=observed,
                op=rule.op,
                value=rule.value,
                passed=ok,
                note=note,
            )
        )
    return GateResult(
        policy=policy.name, passed=all(r.passed for r in results), results=results
    )


def render(result: GateResult) -> str:
    lines = [f"Gate policy: {result.policy}"]
    for r in sorted(result.results, key=lambda r: (r.kind != "hard", r.id)):
        mark = "PASS" if r.passed else "FAIL"
        sym = ">=" if r.op == "min" else "<="
        lines.append(
            f"  {mark} [{r.kind}] {r.id}\n"
            f"       {r.metric} = {r.observed:.4g} (required {sym} {r.value:g})"
        )
        if r.note:
            lines += textwrap.wrap(
                f"note: {r.note}",
                width=72,
                initial_indent=" " * 7,
                subsequent_indent=" " * 13,
            )
    lines.append("RESULT: " + ("PASS" if result.passed else "BLOCKED"))
    return "\n".join(lines)
