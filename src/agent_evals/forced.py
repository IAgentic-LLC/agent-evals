"""The answers that follow three empty searches, and my reading of them (chapter 14).

A run that searched the runbook three times in a row and found nothing must still
answer the customer. Chapter 10 counted 28 of these across four recorded passes. This
module finds them again, joins them to my hand readings, and tests three plain-text
checks against those readings.
"""

import json
import re
from collections import Counter
from pathlib import Path

from agent_evals.schema import Trace
from agent_evals.trajectory import empty_rounds_in_a_row

LABELS = ("gap_stated", "false_action_claim", "claims_source_it_lacks")
GROUPS = {
    "unchanged": ("triage-heldout-v1-calls", "triage-heldout-v1-calls-2"),
    "stall guard": ("triage-heldout-v1-guard", "triage-heldout-v1-guard-2"),
}
# Plain-text proxies for each reading. They look for words and cannot tell a claim
# from a denial, which is why they are measured against the readings.
PROXIES = {
    "gap_stated": re.compile(
        r"could not (?:check|find)|couldn't check|unable to|was unable|not able to"
        r"|no matching",
        re.IGNORECASE,
    ),
    "false_action_claim": re.compile(
        r"\b(?:I have|we have|I will|we will|I'll)\b[^.\n]{0,40}"
        r"\b(?:logged|forwarded|inspect|escalate|ensure)",
        re.IGNORECASE,
    ),
    "claims_source_it_lacks": re.compile(
        r"\b(?:searched|attempted|checked|looked up|query|queried)\b[^.\n]{0,90}"
        r"\b(?:logs?|status|health|database|diagnostic)",
        re.IGNORECASE,
    ),
}


def forced_answers(runs: dict[str, list[Trace]]) -> list[tuple[str, Trace]]:
    """Runs that ended in an answer after three empty rounds in a row."""
    return [
        (run, t)
        for run, traces in runs.items()
        for t in traces
        if t.error is None and t.answer.strip() and empty_rounds_in_a_row(t) >= 3
    ]


def load_readings(path: str | Path) -> dict[tuple[str, str], dict]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    return {(r["run"], r["case_id"]): r for r in rows}


def _group(run: str) -> str:
    return next(g for g, runs in GROUPS.items() if run in runs)


def render_readings(
    forced: list[tuple[str, Trace]], readings: dict[tuple[str, str], dict]
) -> str:
    count: dict[str, Counter] = {g: Counter() for g in GROUPS}
    for run, t in forced:
        row = readings[(run, t.case_id)]
        group = _group(run)
        count[group]["answers"] += 1
        for label in LABELS:
            count[group][label] += bool(row[label])
    head = (
        f"{'':<24}{'answers':>8}{'gap stated':>12}{'false action':>14}"
        f"{'source it lacks':>17}"
    )
    lines = ["Forced answers, read by hand", "", head]
    for group, c in count.items():
        n = c["answers"]
        cells = [f"{c[label]} of {n}" for label in LABELS]
        lines.append(f"{group:<24}{n:>8}{cells[0]:>12}{cells[1]:>14}{cells[2]:>17}")
    return "\n".join(lines) + "\n"


def render_proxies(
    forced: list[tuple[str, Trace]], readings: dict[tuple[str, str], dict]
) -> str:
    lines = [
        f"{'reading':<24}{'yes':>5}{'flagged':>9}{'false alarm':>13}{'missed':>8}",
    ]
    for label in LABELS:
        yes = flagged = false_alarm = missed = 0
        for run, t in forced:
            truth = bool(readings[(run, t.case_id)][label])
            hit = PROXIES[label].search(t.answer) is not None
            yes += truth
            flagged += hit
            false_alarm += hit and not truth
            missed += truth and not hit
        lines.append(f"{label:<24}{yes:>5}{flagged:>9}{false_alarm:>13}{missed:>8}")
    return "\n".join(lines) + "\n"
