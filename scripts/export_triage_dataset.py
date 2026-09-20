"""Write the six-ticket set from Book 3 as a versioned dataset (datasets/triage_book3_six.jsonl).

The cases are the ones in triage_app.evaluation.GOLDEN_TICKETS, unchanged, with their
expected specialist and any forbidden action carried as an explicit invariant.
"""

from pathlib import Path

from triage_app.evaluation import GOLDEN_TICKETS

from agent_evals.schema import EvalCase

OUT = Path(__file__).resolve().parents[1] / "datasets" / "triage_book3_six.jsonl"


def main() -> None:
    lines = []
    for g in GOLDEN_TICKETS:
        t = g.ticket
        adversarial = bool(g.forbidden_actions)
        case = EvalCase(
            case_id=t.ticket_id,
            input=t.model_dump(),
            expected={"handled_by": g.expected_handled_by},
            invariants={"forbidden_actions": list(g.forbidden_actions)}
            if adversarial
            else {},
            slices={
                "adversarial": "yes" if adversarial else "no",
                "miscategorized": "yes"
                if g.expected_handled_by != t.category
                else "no",
            },
            provenance="Book 3 (Production AI Products), chapter 23, triage_app.evaluation.GOLDEN_TICKETS",
        )
        lines.append(case.model_dump_json())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf8")
    print(f"wrote {len(lines)} cases to {OUT}")


if __name__ == "__main__":
    main()
