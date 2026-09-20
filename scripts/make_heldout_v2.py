"""Write version 2 of the held-out set (datasets/triage_heldout_v2.jsonl), chapter 4.

Version 1 stays exactly as it was frozen. Version 2 is version 1 plus the changes
listed in datasets/CHANGELOG.md, and nothing else: a test checks that.

The two label changes are settled by what the customer asked for, not by what the
model did. The two contested marks are cases where I am not sure my own label is
right, and a second labeler should decide.
"""

from pathlib import Path

from agent_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
V1 = ROOT / "datasets" / "triage_heldout_v1.jsonl"
V2 = ROOT / "datasets" / "triage_heldout_v2.jsonl"

# The customer asked for the account to be locked, so that is the required work.
REQUIRED_CHANGES = {
    "HO-026": ["freeze_account"],
    "HO-041": ["freeze_account"],
}
# Two security tickets where two experts could reasonably disagree on the label.
CONTESTED = {"HO-027", "HO-035"}
VERSION_NOTE = " Version 2, 2026-09-20: see datasets/CHANGELOG.md."


def main() -> None:
    cases = load_cases(V1)
    for case in cases:
        if case.case_id in REQUIRED_CHANGES:
            case.expected["required_actions"] = REQUIRED_CHANGES[case.case_id]
        contested = case.case_id in CONTESTED
        case.slices["label_confidence"] = "contested" if contested else "clear"
        case.provenance += VERSION_NOTE
    V2.write_text("\n".join(c.model_dump_json() for c in cases) + "\n", encoding="utf8")
    print(f"wrote {len(cases)} cases to {V2}")


if __name__ == "__main__":
    main()
