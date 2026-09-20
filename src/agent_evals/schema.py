"""The small set of records everything else is built on.

An EvalCase says what to run and what counts as correct. A Trace records what
happened. A Scorecard is the profile of measures for one run: never a single number.
"""

from typing import Any

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    case_id: str
    input: dict[str, Any]
    expected: dict[str, Any]
    # Hard constraints: things that must never happen, whatever the quality score says.
    invariants: dict[str, Any] = Field(default_factory=dict)
    slices: dict[str, str] = Field(default_factory=dict)
    provenance: str = ""
    # "dev" cases may be used to build and tune the system. "test" cases never are.
    split: str = ""
    # A marker that lets training-data filters recognise and exclude this data.
    canary: str = ""


class Trace(BaseModel):
    case_id: str
    trial: int = 1
    adapter: str
    handled_by: str | None = None
    answer: str = ""
    actions_taken: list[dict[str, Any]] = Field(default_factory=list)
    error: str | None = None
    latency_s: float | None = None
    note: str = ""


class Grade(BaseModel):
    case_id: str
    trial: int
    routing_correct: bool
    required_actions_missing: list[str]
    forbidden_actions_taken: list[str]

    @property
    def required_actions_met(self) -> bool:
        return not self.required_actions_missing


class Interval(BaseModel):
    low: float
    high: float
    confidence: float = 0.95


class Scorecard(BaseModel):
    dataset: str
    run: str
    adapters: list[str]
    cases: int
    trials: int
    observations: int
    routing_successes: int
    routing_rate: float
    routing_interval: Interval
    actions_successes: int
    actions_rate: float
    actions_interval: Interval
    invariant_violations: int
    invariant_violation_rate_interval: Interval
    violated_cases: list[str]
    errors: int
    latency_median_s: float | None = None
    latency_p95_s: float | None = None
