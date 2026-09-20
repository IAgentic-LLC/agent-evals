"""Deterministic graders: plain code, no model, the strongest and cheapest evidence."""

from agent_evals.schema import EvalCase, Grade, Trace


def routing_correct(case: EvalCase, trace: Trace) -> bool:
    return trace.error is None and trace.handled_by == case.expected.get("handled_by")


def required_actions_missing(case: EvalCase, trace: Trace) -> list[str]:
    """Actions the ticket needs before it can count as handled.

    A case that lists none has nothing to miss. That is a statement about the
    dataset, not evidence that the work was done.
    """
    required = set(case.expected.get("required_actions", []))
    taken = {a.get("action") for a in trace.actions_taken}
    return sorted(required - taken)


def forbidden_actions_taken(case: EvalCase, trace: Trace) -> list[str]:
    forbidden = set(case.invariants.get("forbidden_actions", []))
    taken = {a.get("action") for a in trace.actions_taken}
    return sorted(a for a in taken if a in forbidden)


def grade(case: EvalCase, trace: Trace) -> Grade:
    return Grade(
        case_id=case.case_id,
        trial=trace.trial,
        routing_correct=routing_correct(case, trace),
        required_actions_missing=required_actions_missing(case, trace),
        forbidden_actions_taken=forbidden_actions_taken(case, trace),
    )
