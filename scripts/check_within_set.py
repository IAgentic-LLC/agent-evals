"""Are any cases in one dataset copies of each other?

Word overlap on every pair of cases, and the subject lines used twice. No key
needed. Word overlap catches copies and light edits, not paraphrases.

    uv run python scripts/check_within_set.py datasets/triage_heldout_v1.jsonl
"""

import sys

from agent_evals.dataset import repeated_subjects, within_set_pairs
from agent_evals.runner import load_cases


def main(path: str) -> None:
    cases = load_cases(path)
    pairs = within_set_pairs(cases)
    print(f"{path}: {len(cases)} cases, {len(pairs)} pairs")
    print(f"most similar pair shares {pairs[0][0]:.1%} of its 3-word phrases")
    for score, a, b in pairs[:5]:
        print(f"  {a} and {b}: {score:.1%}")
    repeats = repeated_subjects(cases)
    print(f"subject lines used more than once: {len(repeats)}")
    for subject, ids in repeats.items():
        print(f"  {subject!r}: {', '.join(ids)}")


if __name__ == "__main__":
    main(sys.argv[1])
