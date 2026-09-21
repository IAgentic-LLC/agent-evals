"""Turning incidents into evals (chapter 25).

An incident is a failure someone saw. It becomes an eval in five steps: write the record,
make a case from the trace, show the case fails on the old product, fix the product, and show
the case passes on the new one and that the gate still lets the rest of the product through.
The ledger holds the record, and `check` re-does the showing from recorded runs, so an incident
cannot be marked fixed without a run that says it is.
"""

import json
import textwrap
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agent_evals import reliability
from agent_evals.answer_graders import acted_on_the_right_customer
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval


class Evidence(BaseModel):
    run: str
    case: str
    trial: int


class Incident(BaseModel):
    id: str
    title: str
    found: str
    chapter: int
    evidence: Evidence
    what_happened: str
    impact: str
    layer: Literal["invariant", "gate", "monitor", "review", "harness"]
    rule: str
    dataset: str | None = None
    cases: list[str] = []
    status: Literal["open", "fixed"]
    before: str | None = None
    after: str | None = None
    fix: str = ""


# ---------------------------------------------------------------------- the rules


def attempted_other_customer(case: EvalCase, trace: Trace) -> bool:
    """A tool call that names a customer id other than the ticket's own, whether or not
    the tool did anything. It is the intent, and `acted_on_the_right_customer` is the deed."""
    own = case.input.get("customer_id")
    return any(
        c.get("arguments", {}).get("customer_id", own) != own for c in trace.tool_calls
    )


def matches(rule: str, case: EvalCase | None, trace: Trace) -> bool:
    """Whether a run is an instance of the incident. The rules are plain code on recorded
    facts: `customer`, `attempt:customer`, `error:PREFIX`, `action:NAME` and
    `usage-missing-thinking`."""
    kind, _, arg = rule.partition(":")
    if kind == "customer":
        return not acted_on_the_right_customer(case, trace)
    if kind == "attempt":
        return attempted_other_customer(case, trace)
    if kind == "error":
        return (trace.error or "").startswith(arg)
    if kind == "action":
        return any(a.get("action") == arg for a in trace.actions_taken)
    if kind == "usage-missing-thinking":
        return "thinking_tokens" not in trace.usage
    raise ValueError(f"unknown incident rule {rule!r}")


# --------------------------------------------------------------------- the ledger

# Zero in 30 runs still allows a rate of about 11 in 100 (95% Wilson interval), and zero in
# 3 allows more than half. A run after the fix with fewer runs than this shows nothing.
MIN_RUNS_AFTER = 30


def load_ledger(root: Path, folder: str = "incidents") -> list[Incident]:
    files = sorted((root / folder).glob("INC-*.yaml"))
    return [
        Incident.model_validate(yaml.safe_load(f.read_text(encoding="utf8")))
        for f in files
    ]


def _runs(root: Path, name: str, only: list[str]):
    traces = read_traces(root / "runs" / name / "traces.jsonl")
    return [t for t in traces if not only or t.case_id in only]


def check(inc: Incident, root: Path) -> list[str]:
    """What is wrong with the incident's record, from the recorded runs. An empty list
    means the record holds: the evidence trace is an instance of the incident, the run
    before the fix shows it, and a fixed incident has a run after the fix that does not."""
    problems: list[str] = []
    cases = {}
    if inc.dataset:
        cases = {c.case_id: c for c in load_cases(root / inc.dataset)}
        missing = [c for c in inc.cases if c not in cases]
        if missing:
            problems.append(f"cases not in {inc.dataset}: {', '.join(missing)}")
    evidence = [
        t
        for t in read_traces(root / "runs" / inc.evidence.run / "traces.jsonl")
        if t.case_id == inc.evidence.case and t.trial == inc.evidence.trial
    ]
    if not evidence:
        problems.append("the evidence trace is not in its run")
    else:
        case = evidence_case(inc, root)
        if not matches(inc.rule, case, evidence[0]):
            problems.append("the evidence trace does not match the rule")
    if inc.before:
        rows = _runs(root, inc.before, inc.cases)
        if not any(matches(inc.rule, cases.get(t.case_id), t) for t in rows):
            problems.append("the run before the fix does not show the incident")
    if inc.status == "fixed":
        if not inc.after:
            problems.append("a fixed incident needs a run after the fix")
        else:
            rows = _runs(root, inc.after, inc.cases)
            bad = sum(matches(inc.rule, cases.get(t.case_id), t) for t in rows)
            if len(rows) < MIN_RUNS_AFTER:
                problems.append(
                    f"the run after the fix has {len(rows)} runs of the cases, "
                    f"and it needs {MIN_RUNS_AFTER}"
                )
            elif bad:
                problems.append(f"the run after the fix still shows it, {bad} runs")
    elif inc.after:
        problems.append("an open incident has no run after a fix")
    return problems


def evidence_case(inc: Incident, root: Path) -> EvalCase | None:
    """The case of the evidence trace, from its own run's dataset."""
    manifest = json.loads(
        (root / "runs" / inc.evidence.run / "manifest.json").read_text(encoding="utf8")
    )
    path = Path(manifest["dataset"]["path"])
    path = path if path.is_absolute() else root / path
    for c in load_cases(path):
        if c.case_id == inc.evidence.case:
            return c
    return None


def render_ledger(rows: list[tuple[Incident, list[str]]]) -> str:
    lines = [f"{'id':<9}{'layer':<11}{'status':<8}{'cases':>6}  {'record':<7}title"]
    for inc, problems in rows:
        title = textwrap.shorten(inc.title, width=35, placeholder="...")
        lines.append(
            f"{inc.id:<9}{inc.layer:<11}{inc.status:<8}{len(inc.cases):>6}  "
            + f"{'ok' if not problems else 'BROKEN':<7}{title}"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------ from a trace


def draft_case(
    case: EvalCase, incident_id: str, run: str, trial: int, new_id: str
) -> dict:
    """A new case from an incident's own ticket: the same input, a new id, and where it
    came from. It is a dev case, because it will shape the fix."""
    data = case.model_dump()
    data["case_id"] = new_id
    data["input"] = {**data["input"], "ticket_id": new_id}
    data["slices"] = {**data["slices"], "incident": incident_id}
    data["provenance"] = (
        f"From incident {incident_id} (run {run}, case {case.case_id}, trial {trial}). "
        "A dev case: it shaped the fix."
    )
    data["split"] = "dev"
    return data


def render_draft(data: dict) -> str:
    lines = [f"case_id: {data['case_id']}", f"split: {data['split']}"]
    for key in ("customer_id", "category", "subject"):
        lines.append(f"{key}: {data['input'][key]}")
    lines += textwrap.wrap(
        "body: " + data["input"]["body"], width=76, subsequent_indent="  "
    )
    expected = data["expected"]
    lines.append(
        f"expected: handled by {expected['handled_by']}, "
        f"required {', '.join(expected['required_actions'])}"
    )
    lines += textwrap.wrap(
        "provenance: " + data["provenance"], width=76, subsequent_indent="  "
    )
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------- the tables


def _pct(k: int, n: int) -> str:
    return f"{100 * k / n:.0f}%"


def render_reproduction(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """How often the incident happens, by kind of ticket: runs that tried another customer id
    and runs that acted on one."""
    groups: dict[str, list[Trace]] = {}
    for t in traces:
        groups.setdefault(cases[t.case_id].slices.get("variant", "all"), []).append(t)
    order = ["original", "shorter", "has-invoice", "other-id", "new-wording"]
    lines = [
        f"{'ticket':<14}{'tickets':>8}{'runs':>6}{'tried':>8}{'acted':>8}{'met':>6}"
    ]
    for name in [*order, "all"]:
        rows = traces if name == "all" else groups.get(name, [])
        if not rows:
            continue
        n = len(rows)
        tickets = len({t.case_id for t in rows})
        tried = sum(attempted_other_customer(cases[t.case_id], t) for t in rows)
        acted = sum(not acted_on_the_right_customer(cases[t.case_id], t) for t in rows)
        met = sum(reliability.success(cases[t.case_id], t) for t in rows)
        lines.append(
            f"{name:<14}{tickets:>8}{n:>6}{tried:>8}{acted:>8}{_pct(met, n):>6}"
        )
    lines.append("tried and acted are runs, from the tool calls and from the actions")
    return "\n".join(lines) + "\n"


def render_fix(conditions: list[tuple[str, dict[str, EvalCase], list[Trace]]]) -> str:
    """The same tickets under each version: runs that tried another id, runs that acted on
    one with the interval on that share, and the met share."""
    lines = [
        f"{'version':<24}{'runs':>6}{'tried':>7}{'acted':>7}{'interval':>13}{'met':>6}"
    ]
    for label, cases, traces in conditions:
        n = len(traces)
        tried = sum(attempted_other_customer(cases[t.case_id], t) for t in traces)
        acted = sum(
            not acted_on_the_right_customer(cases[t.case_id], t) for t in traces
        )
        met = sum(reliability.success(cases[t.case_id], t) for t in traces)
        low, high = wilson_interval(acted, n)
        lines.append(
            f"{label:<24}{n:>6}{tried:>7}{acted:>7}"
            + f"{f'{100 * low:.1f} to {100 * high:.1f}%':>13}{_pct(met, n):>6}"
        )
    lines.append("interval is the 95% Wilson interval on the share of runs that acted")
    return "\n".join(lines) + "\n"


def render_reach(rows: list[tuple[str, int, int, int]]) -> str:
    """The chance that a check of a given size sees the incident at least once, if it
    happens at the rate seen: (label, runs that showed it, runs seen, runs in the check)."""
    lines = [f"{'check':<28}{'rate seen':>10}{'runs':>6}{'sees it':>9}"]
    for label, k, seen, size in rows:
        p = k / seen
        lines.append(
            f"{label:<28}{f'{k} in {seen}':>10}{size:>6}{f'{100 * (1 - (1 - p) ** size):.0f}%':>9}"
        )
    lines.append("chance of at least one instance, at the rate seen")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- the queue

TOOL_LOOP_EVERY = 10
OTHER_EVERY = 50
MINUTES_PER_TRACE = 3


def render_queue(runs: list[tuple[dict[str, EvalCase], list[Trace]]]) -> str:
    """Which runs a person reads, from recorded runs. All that broke the customer check, the
    first handoff loop of each ticket, every tenth tool loop, and every fiftieth of the rest,
    in a fixed order so the queue is the same every time."""
    counts = {
        "customer check": [0, 0],
        "handoff loop": [0, 0],
        "tool loop": [0, 0],
        "the rest": [0, 0],
    }
    seen_handoff: set[tuple[int, str]] = set()
    loops = other = 0
    for i, (cases, traces) in enumerate(runs):
        for t in traces:
            case = cases[t.case_id]
            if not acted_on_the_right_customer(case, t):
                counts["customer check"][0] += 1
                counts["customer check"][1] += 1
            elif (t.error or "").startswith("HandoffLoop"):
                counts["handoff loop"][0] += 1
                if (i, t.case_id) not in seen_handoff:
                    seen_handoff.add((i, t.case_id))
                    counts["handoff loop"][1] += 1
            elif t.error:
                counts["tool loop"][0] += 1
                loops += 1
                counts["tool loop"][1] += loops % TOOL_LOOP_EVERY == 0
            else:
                counts["the rest"][0] += 1
                other += 1
                counts["the rest"][1] += other % OTHER_EVERY == 0
    lines = [f"{'why a person reads it':<24}{'runs':>6}{'read':>6}"]
    total = 0
    for name, (n, read) in counts.items():
        lines.append(f"{name:<24}{n:>6}{read:>6}")
        total += read
    hours = total * MINUTES_PER_TRACE / 60
    lines.append(f"{'all':<24}{sum(v[0] for v in counts.values()):>6}{total:>6}")
    lines.append(f"about {hours:.1f} hours at {MINUTES_PER_TRACE} minutes a trace")
    return "\n".join(lines) + "\n"
