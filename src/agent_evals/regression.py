"""Regression gates: is a candidate run worse than a baseline run? (chapter 23)

Chapter 1's gate reads one run and compares it with a fixed number. A change to a product
is a different question: is this run worse than the last good one? Two runs of the same
product differ by luck alone, so the answer is a paired comparison with an interval, and
a rule for what to do when the interval does not settle it.

A check has one of three kinds. A hard check counts violations and allows none. A
regression check compares the met rate of the two runs ticket by ticket. A budget check
compares cost or latency and only warns. The risk tier of the change, from Book 3's
release gate, decides which checks apply.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel

from agent_evals import compare, cost, reliability
from agent_evals.answer_graders import acted_on_the_right_customer
from agent_evals.graders import forbidden_actions_taken
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import percentile

Status = Literal["pass", "block", "hold", "warn", "skip"]
TIERS = ("low", "medium", "high")


class Requirements(BaseModel):
    same_dataset: bool = True
    same_model: bool = True
    min_tickets: int = 30


class Check(BaseModel):
    id: str
    kind: Literal["hard", "regression", "budget"]
    tiers: list[str] = ["high"]
    # hard
    check: Literal["customer", "forbidden"] | None = None
    # regression
    method: Literal["interval", "sign", "point"] = "interval"
    margin_points: float = 0.0
    alpha: float = 0.05
    on_inconclusive: Literal["hold", "block", "pass"] = "hold"
    # budget
    budget: Literal["cost_per_success", "latency_p95"] | None = None
    max_ratio: float = 1.5


class Policy(BaseModel):
    name: str
    require: Requirements = Requirements()
    checks: list[Check]


@dataclass
class Finding:
    id: str
    kind: str
    status: Status
    lines: list[str] = field(default_factory=list)


@dataclass
class Report:
    policy: str
    tier: str
    header: list[str]
    findings: list[Finding]

    @property
    def status(self) -> str:
        """BLOCK beats HOLD beats WARN beats PASS."""
        found = {f.status for f in self.findings}
        for status in ("block", "hold", "warn"):
            if status in found:
                return status.upper()
        return "PASS"

    @property
    def exit_code(self) -> int:
        return {"BLOCK": 1, "HOLD": 2}.get(self.status, 0)


def load_policy(path: str | Path) -> Policy:
    return Policy.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf8")))


def tier_for_files(files: list[str]) -> str:
    """The risk tier of a change, from the release gate Book 3 built."""
    from triage_app.release_gate import classify_change

    return str(classify_change(files))


# ------------------------------------------------------------------ decisions


def ticket_differences(
    cases: dict[str, EvalCase], baseline: list[Trace], candidate: list[Trace]
) -> list[float]:
    """For each ticket, the candidate's share of runs that met the required actions
    minus the baseline's. Positive means the candidate did better on that ticket."""
    shares = []
    for traces in (baseline, candidate):
        by_ticket: dict[str, list[bool]] = {}
        for t in traces:
            by_ticket.setdefault(t.case_id, []).append(
                reliability.success(cases[t.case_id], t)
            )
        shares.append({k: sum(v) / len(v) for k, v in by_ticket.items()})
    base, cand = shares
    return [cand[k] - base[k] for k in sorted(set(base) & set(cand))]


def decide(
    diffs: list[float],
    method: str,
    margin_points: float,
    alpha: float = 0.05,
    interval: tuple[float, float] | None = None,
) -> tuple[str, str]:
    """(verdict, reason) for one regression check: pass, block or inconclusive.

    `interval` is the 95% interval for the mean difference, drawn by redrawing whole
    tickets. It does not depend on the margin, so callers that try several margins can
    draw it once."""
    mean = sum(diffs) / len(diffs)
    margin = margin_points / 100
    if method == "point":
        if mean < -margin:
            return "block", f"the met rate fell by more than {margin_points:g} points"
        return (
            "pass",
            f"the met rate did not fall by more than {margin_points:g} points",
        )
    if method == "sign":
        better = sum(d > 0 for d in diffs)
        worse = sum(d < 0 for d in diffs)
        p = compare.exact_sign_test(better, worse)
        if worse > better and p < alpha:
            return "block", f"significantly worse (p = {p:.3f})"
        return "pass", f"not significantly worse (p = {p:.3f})"
    low, high = interval or compare.cluster_interval(compare.mean, diffs)
    if low >= -margin:
        return "pass", f"no drop of {margin_points:g} points or more is plausible"
    if high < -margin:
        return "block", f"a drop of more than {margin_points:g} points is shown"
    return (
        "inconclusive",
        f"a drop of {margin_points:g} points or more is not ruled out",
    )


# ------------------------------------------------------------------ the report


def _manifest(run: Path) -> dict:
    return json.loads((run / "manifest.json").read_text(encoding="utf8"))


def _problems(pol: Requirements, base: dict, cand: dict, tickets: int) -> list[str]:
    problems = []
    if pol.same_dataset and base["dataset"]["sha256"] != cand["dataset"]["sha256"]:
        problems.append("the two runs used different dataset files")
    if pol.same_model:
        models = [m.get("model", {}).get("model_id") for m in (base, cand)]
        if None in models:
            problems.append("a run did not record which model answered")
        elif models[0] != models[1]:
            problems.append(f"the models differ: {models[0]} and {models[1]}")
    if tickets < pol.min_tickets:
        problems.append(f"only {tickets} tickets; the policy needs {pol.min_tickets}")
    return problems


def _met(cases: dict[str, EvalCase], traces: list[Trace]) -> float:
    return sum(reliability.success(cases[t.case_id], t) for t in traces) / len(traces)


def _hard(check: Check, cases: dict[str, EvalCase], base, cand) -> Finding:
    def broken(traces: list[Trace]) -> list[Trace]:
        if check.check == "customer":
            return [
                t
                for t in traces
                if not acted_on_the_right_customer(cases[t.case_id], t)
            ]
        return [t for t in traces if forbidden_actions_taken(cases[t.case_id], t)]

    what = {
        "customer": "runs act on another customer",
        "forbidden": "runs take a forbidden action",
    }[check.check or "customer"]
    bad = broken(cand)
    lines = [f"candidate {len(bad)} {what}, baseline {len(broken(base))}"]
    for t in bad[:3]:
        lines.append(f"{t.case_id} run {t.trial}")
    return Finding(check.id, "hard", "block" if bad else "pass", lines)


def _regression(check: Check, cases: dict[str, EvalCase], base, cand) -> Finding:
    diffs = ticket_differences(cases, base, cand)
    low, high = compare.cluster_interval(compare.mean, diffs)
    mean = sum(diffs) / len(diffs)
    verdict, reason = decide(
        diffs, check.method, check.margin_points, check.alpha, (low, high)
    )
    status: Status = {"pass": "pass", "block": "block"}.get(verdict, "hold")
    if verdict == "inconclusive":
        status = {"hold": "hold", "block": "block", "pass": "pass"}[
            check.on_inconclusive
        ]
    lines = [
        f"met {100 * _met(cases, base):.1f}% -> {100 * _met(cases, cand):.1f}%, "
        + f"difference {100 * mean:+.1f} points",
        f"95% interval {100 * low:+.1f} to {100 * high:+.1f}; margin "
        + f"-{check.margin_points:g}",
        reason,
    ]
    if verdict == "inconclusive":
        lines.append(f"policy for an unsettled result: {check.on_inconclusive}")
    return Finding(check.id, "regression", status, lines)


def _budget(check: Check, cases, base, cand, base_run, cand_run, prices) -> Finding:
    if check.budget == "latency_p95":
        times = [
            [t.latency_s for t in traces if t.latency_s is not None]
            for traces in (base, cand)
        ]
        if not all(times):
            return Finding(check.id, "budget", "skip", ["no latency recorded"])
        b, c = (percentile(v, 0.95) for v in times)
        label = "p95 latency"
        values = (f"{b:.1f} s", f"{c:.1f} s")
    else:
        try:
            per_success = []
            for traces, run in ((base, base_run), (cand, cand_run)):
                price = prices[_manifest(run)["model"]["model_id"]]
                total = sum(cost.run_cost(t, price) for t in traces)
                per_success.append(
                    total
                    / max(
                        1, sum(reliability.success(cases[t.case_id], t) for t in traces)
                    )
                )
        except (KeyError, ValueError):
            return Finding(check.id, "budget", "skip", ["no tokens or prices recorded"])
        b, c = per_success
        label = "cost per success"
        values = (f"${b:.4f}", f"${c:.4f}")
    ratio = c / b
    status: Status = "warn" if ratio > check.max_ratio else "pass"
    return Finding(
        check.id,
        "budget",
        status,
        [
            f"{label} {values[0]} -> {values[1]}, {ratio:.2f} times (limit {check.max_ratio:g})"
        ],
    )


def evaluate(
    policy: Policy,
    baseline_run: str | Path,
    candidate_run: str | Path,
    dataset: str | Path,
    tier: str = "high",
    prices: dict[str, tuple[float, float]] | None = None,
) -> Report:
    base_run, cand_run = Path(baseline_run), Path(candidate_run)
    cases = {c.case_id: c for c in load_cases(dataset)}
    base = read_traces(base_run / "traces.jsonl")
    cand = read_traces(cand_run / "traces.jsonl")
    bm, cm = _manifest(base_run), _manifest(cand_run)
    tickets = len({t.case_id for t in base} & {t.case_id for t in cand})
    header = [
        f"Risk tier: {tier.upper()}",
        f"Baseline {base_run.name}, candidate {cand_run.name}",
        f"{tickets} tickets, {bm['trials']} and {cm['trials']} trials, dataset "
        + f"{cm['dataset']['sha256'][:8]}",
    ]
    findings: list[Finding] = []
    problems = _problems(policy.require, bm, cm, tickets)
    if problems:
        findings.append(
            Finding("comparison-is-valid", "requirement", "block", problems)
        )
        return Report(policy.name, tier, header, findings)
    applies = [c for c in policy.checks if tier in c.tiers]
    if not applies:
        findings.append(
            Finding("no-checks-at-this-tier", "requirement", "pass", ["nothing to run"])
        )
    for check in applies:
        if check.kind == "hard":
            findings.append(_hard(check, cases, base, cand))
        elif check.kind == "regression":
            findings.append(_regression(check, cases, base, cand))
        else:
            findings.append(
                _budget(check, cases, base, cand, base_run, cand_run, prices or {})
            )
    return Report(policy.name, tier, header, findings)


def render(report: Report) -> str:
    lines = [f"Change gate: {report.policy}", *report.header, ""]
    order = {"hard": 0, "regression": 1, "budget": 2, "requirement": 0}
    for f in sorted(report.findings, key=lambda f: (order[f.kind], f.id)):
        lines.append(f"  {f.status.upper():<5} [{f.kind}] {f.id}")
        lines += [f"        {line}" for line in f.lines]
    lines.append(f"RESULT: {report.status}")
    return "\n".join(lines) + "\n"


def render_markdown(report: Report) -> str:
    """The same report as a table, for a job summary."""
    rows = ["| Check | Kind | Status | Detail |", "|:--|:--|:--|:--|"]
    for f in report.findings:
        detail = "<br>".join(f.lines).replace("|", "/")
        rows.append(f"| {f.id} | {f.kind} | {f.status.upper()} | {detail} |")
    head = f"### Change gate: {report.policy}, {report.status}\n\n" + "  \n".join(
        report.header
    )
    return head + "\n\n" + "\n".join(rows) + "\n"
