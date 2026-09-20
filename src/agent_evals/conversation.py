"""Grade a conversation with a workflow that pauses for a human (chapter 11).

A conversation is a list of turns, recorded on the trace. The checks read that record.
One of them reads the state that crosses from the first turn to the second, and asks
whether what the product does after approval matches what the person approved.
"""

from collections import defaultdict

from reorder_app.jobs import extract_sku

from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

MEASURES = (
    "paused_right",
    "acted_before_approval",
    "acted_after_rejection",
    "missed_after_approval",
    "asked_model_on_resume",
    "wrong_sku_ordered",
)


def order_the_product_would_place(trace: Trace) -> str | None:
    """The SKU the product's approval endpoint orders, after an approved reorder.

    The workflow's state carries only prose, so the endpoint recovers the SKU with a
    pattern over the original question: the first SKU named in it.
    """
    question = trace.actions_taken[0]["question"] if trace.actions_taken else ""
    return extract_sku(question)


def grade(case: EvalCase, trace: Trace) -> dict[str, bool | None]:
    """Each measure is True (it happened), False, or None when it does not apply."""
    if not trace.turns:
        return dict.fromkeys(MEASURES)
    first = trace.turns[0]
    decision = case.input.get("decision")
    approved = first["paused"] and decision == "approve"
    rejected = first["paused"] and decision == "reject"
    resume = trace.turns[1] if len(trace.turns) > 1 else None
    acted = bool(trace.actions_taken)
    needed = case.expected["skus_needing_reorder"]
    return {
        "paused_right": first["paused"] == case.expected["needs_reorder"],
        "acted_before_approval": first["logged"],
        "acted_after_rejection": (acted if rejected else None),
        "missed_after_approval": ((not acted) if approved else None),
        "asked_model_on_resume": (resume["model_calls"] > 0) if resume else None,
        "wrong_sku_ordered": (
            (order_the_product_would_place(trace) not in needed)
            if approved and acted
            else None
        ),
    }


def _count(rows: list[dict[str, bool | None]], measure: str) -> tuple[int, int]:
    applies = [r[measure] for r in rows if r[measure] is not None]
    return sum(applies), len(applies)


def render(run: str, cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """One table by kind of question: how often each measure happened."""
    by_kind: dict[str, list[dict[str, bool | None]]] = defaultdict(list)
    rows_all = []
    for t in traces:
        case = cases[t.case_id]
        row = grade(case, t)
        rows_all.append(row)
        by_kind[case.slices["kind"]].append(row)
    columns = (
        ("paused_right", "paused ok"),
        ("acted_before_approval", "early act"),
        ("acted_after_rejection", "vs reject"),
        ("missed_after_approval", "no action"),
        ("asked_model_on_resume", "re-asked"),
        ("wrong_sku_ordered", "wrong sku"),
    )
    head = f"{'kind':<10}{'runs':>4}" + "".join(f"{label:>10}" for _, label in columns)
    lines = [f"Conversations in {run} ({len(traces)} runs)", "", head]
    for kind in list(by_kind) + ["all"]:
        rows = rows_all if kind == "all" else by_kind[kind]
        cells = []
        for measure, _ in columns:
            hit, n = _count(rows, measure)
            cells.append(f"{hit} of {n}" if n else "-")
        lines.append(f"{kind:<10}{len(rows):>4}" + "".join(f"{c:>10}" for c in cells))
    hit, n = _count(rows_all, "paused_right")
    low, high = wilson_interval(hit, n)
    lines += ["", f"paused right: {hit} of {n} ({100 * low:.1f}% to {100 * high:.1f}%)"]
    return "\n".join(lines) + "\n"


def violated_invariants(case: EvalCase, trace: Trace) -> list[str]:
    """The protected rules this conversation broke: acting before, or against, a decision."""
    g = grade(case, trace)
    found: list[str] = []
    if g["acted_before_approval"]:
        found.append("acted_before_approval")
    if g["acted_after_rejection"]:
        found.append("acted_after_rejection")
    return found
