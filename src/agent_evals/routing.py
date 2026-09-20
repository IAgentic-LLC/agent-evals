"""Routing and handoffs in the triage product (chapter 20).

A ticket arrives with a category. The product starts it with the specialist for that
category. A specialist that thinks the ticket belongs elsewhere calls `request_handoff`
with a target, and the router passes the ticket on. A ticket that is passed back to a
specialist that already had it ends in `HandoffLoopDetected`.

This module separates four things that the single score "routed correctly" mixes:
which specialist finished, whether a handoff was asked for when the category was wrong,
whether one was asked for when the category was right, and where the loops are.
"""

import textwrap
from collections import Counter

from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

SPECIALISTS = ("billing", "security", "technical")


def _span(interval: tuple[float, float]) -> str:
    return f"{100 * interval[0]:.0f}-{100 * interval[1]:.0f}%"


def where(case: EvalCase, trace: Trace) -> str:
    """What became of a run: it finished on the right specialist, on a wrong one, or
    ended in an error before finishing."""
    if trace.error is not None or trace.handled_by is None:
        return "error"
    return "right" if trace.handled_by == case.expected["handled_by"] else "wrong"


def handoffs(trace: Trace) -> list[dict]:
    """The handoff requests a run made, in order."""
    return [c for c in trace.tool_calls if c.get("name") == "request_handoff"]


def targets(trace: Trace) -> list[str]:
    return [h.get("arguments", {}).get("target_category", "?") for h in handoffs(trace)]


def render_specialists(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """One row per specialist a ticket should end with: what happened to its runs."""
    lines = [
        f"{'should end with':<17}{'tickets':>8}{'runs':>6}{'right':>7}"
        + f"{'wrong':>7}{'error':>7}{'always right':>14}{'95% interval':>14}"
    ]
    for name in SPECIALISTS:
        mine = [t for t in traces if cases[t.case_id].expected["handled_by"] == name]
        tickets = sorted({t.case_id for t in mine})
        counts = Counter(where(cases[t.case_id], t) for t in mine)
        always = sum(
            all(
                where(cases[t.case_id], t) == "right"
                for t in mine
                if t.case_id == ticket
            )
            for ticket in tickets
        )
        span = _span(wilson_interval(always, len(tickets)))
        lines.append(
            f"{name:<17}{len(tickets):>8}{len(mine):>6}{counts['right']:>7}"
            f"{counts['wrong']:>7}{counts['error']:>7}"
            f"{f'{always} of {len(tickets)}':>14}{span:>14}"
        )
    return "\n".join(lines) + "\n"


def render_handoffs(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """Handoff requests, split by whether the ticket's category was right."""
    lines = [
        f"{'':<24}{'tickets':>8}{'runs':>6}{'asked':>7}{'right target':>14}{'error':>7}"
    ]
    for label, wrong in (("category wrong", True), ("category right", False)):
        mine = [
            t
            for t in traces
            if (
                cases[t.case_id].input["category"]
                != cases[t.case_id].expected["handled_by"]
            )
            == wrong
        ]
        asked = [t for t in mine if handoffs(t)]
        right = [
            t for t in asked if targets(t)[0] == cases[t.case_id].expected["handled_by"]
        ]
        errors = sum(t.error is not None for t in asked)
        lines.append(
            f"{label:<24}{len({t.case_id for t in mine}):>8}{len(mine):>6}"
            f"{len(asked):>7}{(len(right) if wrong else '-'):>14}{errors:>7}"
        )
    return "\n".join(lines) + "\n"


def render_loops(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """Every ticket that ended in a handoff loop: its category, what it should end
    with, and the targets it was sent to."""
    rows: dict[str, list[Trace]] = {}
    for t in traces:
        if t.error and t.error.startswith("HandoffLoopDetected"):
            rows.setdefault(t.case_id, []).append(t)
    lines = [f"{'ticket':<8}{'category':<11}{'belongs to':<12}{'loops':>6}  sent to"]
    for case_id in sorted(rows):
        case = cases[case_id]
        paths = Counter(" then ".join(targets(t)) for t in rows[case_id])
        text = "; ".join(f"{p} ({n})" for p, n in paths.most_common())
        lines.append(
            f"{case_id:<8}{case.input['category']:<11}{case.expected['handled_by']:<12}"
            f"{len(rows[case_id]):>6}  {text}"
        )
    return "\n".join(lines) + "\n"


def render_reasons(
    cases: dict[str, EvalCase], traces: list[Trace], tickets: list[str]
) -> str:
    """The first reason each of these tickets was given for a handoff."""
    lines = []
    for case_id in tickets:
        for t in traces:
            if t.case_id == case_id and handoffs(t):
                h = handoffs(t)[0]["arguments"]
                head = f"{case_id} (category {cases[case_id].input['category']}) -> {h.get('target_category')}:"
                lines.append(head)
                lines += [
                    "  " + x
                    for x in textwrap.wrap(str(h.get("reason", "")), width=74)[:3]
                ]
                break
    return "\n".join(lines) + "\n"
