"""Three ways to ask "did the refund happen?", to compare (chapter 6).

  mentions_action    naive prose: the answer contains a word about the action
  claims_action_done careful prose: the answer says it was done, in a completed
                     tense, not negated, in English, Spanish or French
  did_action         state: the record of side effects contains the action

The first two read what the agent said. The third reads what happened.
"""

import re
from collections import Counter
from collections.abc import Iterable

from agent_evals.answer_graders import Grader
from agent_evals.schema import Trace

_MENTION = {
    "issue_refund": r"refund|reimburs|credit(?:ed)?\b|reembols|rembours",
    "freeze_account": r"froz|freez|congel|gel[eé]",
    "escalate_to_oncall": r"escalat|on-?call|escalad|transmis",
    "restart_service": r"restart|reinici|red[eé]marr",
}
# The verb words for each action, in English, Spanish and French.
_DONE = {
    "issue_refund": (
        r"refund(?:ed)?|issued a credit|credited",
        r"reembols\w*|devoluci\w*",
        r"rembours\w*",
    ),
    "freeze_account": (
        r"froze\w*|frozen|freez\w*",
        r"congel\w*",
        r"gel\w*|bloqu\w*",
    ),
    "escalate_to_oncall": (
        r"escalat\w+",
        r"escal\w+",
        r"transmis\w*|escalad\w*",
    ),
    "restart_service": (r"restart\w*", r"reinici\w*", r"red[eé]marr\w*"),
}
_GAP = r"[^.?!\n]{0,60}?"


def _claim_pattern(action: str) -> re.Pattern:
    en, es, fr = _DONE[action]
    english = (
        r"\b(?:i have|i've|we have|we've|has been|have been|was|were|is now|been)\b"
        rf"(?!\s+not\b){_GAP}\b(?:{en})"
    )
    spanish = rf"\b(?:hemos|he|ya|se ha|ha sido)\b{_GAP}(?:{es})"
    french = rf"\b(?:j'ai|nous avons|a été|ont été)\b{_GAP}(?:{fr})"
    return re.compile(f"{english}|{spanish}|{french}", re.IGNORECASE)


def mentions_action(action: str) -> Grader:
    pattern = re.compile(_MENTION[action], re.IGNORECASE)
    return lambda case, trace: bool(pattern.search(trace.answer))


def claims_action_done(action: str) -> Grader:
    pattern = _claim_pattern(action)
    return lambda case, trace: bool(pattern.search(trace.answer))


def did_action(action: str) -> Grader:
    return lambda case, trace: any(
        a.get("action") == action for a in trace.actions_taken
    )


ACTIONS = tuple(_MENTION)


def compare(prose: Grader, state: Grader, traces: Iterable[Trace]) -> Counter:
    """Count agreement between a prose grader and the state: both, prose only, state
    only, neither. Traces with an error or no answer are left out: there is no prose."""
    counts: Counter = Counter()
    for trace in traces:
        if trace.error or not trace.answer.strip():
            continue
        said, did = prose(None, trace), state(None, trace)
        counts[
            "both"
            if said and did
            else "prose_only"
            if said
            else "state_only"
            if did
            else "neither"
        ] += 1
    return counts
