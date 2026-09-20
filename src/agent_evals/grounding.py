"""Check what an answer says against what was retrieved (chapter 13).

Plain code, so the same answer always gets the same verdict. Each check looks at one
way an answer can fail to stand on its context, and none of them reads meaning:

- invalid_citation: the answer cites a package that was not among the retrieved three.
- no_citation: a relevant package was retrieved, and the answer cites nothing.
- outside_name: the answer names another package in the index that appears neither
  in the question nor in the retrieved text, and is not itself one of the three.
- new_number: the answer gives a version-like number that appears neither in the
  question nor in the retrieved text.
- ignored_context: a counterfactual question, and the answer lacks the fact that the
  edited summary states.
- error: the run failed before an answer was parsed.

They find candidates. Whether an answer really goes beyond its context is a reading
task, and the chapter reads a sample by hand to see how far these checks can be trusted.
"""

import re
from collections import Counter
from typing import Any

from agent_evals.retrieval import hit_at_k
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

FLAGS = (
    "invalid_citation",
    "no_citation",
    "outside_name",
    "new_number",
    "ignored_context",
    "error",
)
VERSION = re.compile(r"\b\d+(?:\.\d+)+\b")
KINDS = ("task", "named", "hard", "none", "fresh", "counterfactual")


def mentions(text: str, name: str) -> bool:
    """Is `name` in `text` as a whole word, ignoring case?"""
    pattern = rf"(?<![\w-]){re.escape(name)}(?![\w-])"
    return re.search(pattern, text, re.IGNORECASE) is not None


def check(case: EvalCase, trace: Trace, names: list[str]) -> dict[str, Any]:
    """The findings for one answer. A flag is True, False, or None when it does not
    apply to this answer."""
    found = {flag: None for flag in FLAGS}
    found["error"] = trace.error is not None
    found["stated"] = []
    found["outside"] = []
    if trace.error is not None:
        return found
    retrieved = [r["name"] for r in trace.retrieved]
    known = {n.casefold() for n in retrieved}
    cited = [c for c in trace.cited if c.strip()]
    found["invalid_citation"] = any(c.casefold() not in known for c in cited)
    relevant = case.expected.get("relevant", [])
    if relevant and hit_at_k(retrieved, relevant, len(retrieved)):
        found["no_citation"] = not cited
    context = " ".join(r["summary"] for r in trace.retrieved)
    seen = f"{case.input['query']} {context} {' '.join(retrieved)}"
    outside = [n for n in names if mentions(trace.answer, n) and not mentions(seen, n)]
    found["outside"] = outside
    found["outside_name"] = bool(outside)
    numbers = sorted(set(VERSION.findall(trace.answer)))
    found["stated"] = numbers
    found["new_number"] = any(n not in seen for n in numbers)
    if "markers" in case.expected:
        text = trace.answer.casefold()
        found["ignored_context"] = not any(
            m.casefold() in text for m in case.expected["markers"]
        )
    return found


def _rate(k: int, n: int) -> str:
    if n == 0:
        return "-"
    return f"{k} of {n}"


def tally(
    cases: dict[str, EvalCase], traces: list[Trace], names: list[str]
) -> dict[str, Counter]:
    """Per kind and overall: how many answers each flag applies to, and how many it
    caught. Keys `runs`, `<flag>` and `<flag>_of`."""
    table: dict[str, Counter] = {k: Counter() for k in (*KINDS, "all")}
    for trace in traces:
        case = cases[trace.case_id]
        kind = case.slices["kind"]
        found = check(case, trace, names)
        for row in (kind, "all"):
            table[row]["runs"] += 1
            for flag in FLAGS:
                if found[flag] is None:
                    continue
                table[row][f"{flag}_of"] += 1
                table[row][flag] += bool(found[flag])
    return table


def render_summary(
    label: str, cases: dict[str, EvalCase], traces: list[Trace], names: list[str]
) -> str:
    table = tally(cases, traces, names)
    head = (
        f"{'kind':<15}{'runs':>5}{'error':>7}{'bad cite':>10}{'no cite':>10}"
        f"{'outside':>10}{'number':>10}{'ignored':>10}"
    )
    lines = [f"Answers in {label} ({len(traces)} runs)", "", head]
    for kind in (*KINDS, "all"):
        row = table[kind]
        if not row["runs"]:
            continue
        cells = [
            row["error"],
            _rate(row["invalid_citation"], row["invalid_citation_of"]),
            _rate(row["no_citation"], row["no_citation_of"]),
            _rate(row["outside_name"], row["outside_name_of"]),
            _rate(row["new_number"], row["new_number_of"]),
            _rate(row["ignored_context"], row["ignored_context_of"]),
        ]
        widths = (7, 10, 10, 10, 10, 10)
        lines.append(
            f"{kind:<15}{row['runs']:>5}"
            + "".join(f"{c:>{w}}" for c, w in zip(cells, widths))
        )
    return "\n".join(lines) + "\n"


def render_bounds(
    cases: dict[str, EvalCase], traces: list[Trace], names: list[str]
) -> str:
    """The count and the 95% interval for each flag over every run it applies to."""
    row = tally(cases, traces, names)["all"]
    lines = [f"{'flag':<18}{'count':>10}   95% interval"]
    for flag in FLAGS:
        n, k = row[f"{flag}_of"], row[flag]
        low, high = wilson_interval(k, n)
        lines.append(
            f"{flag:<18}{_rate(k, n):>10}   {100 * low:.1f}% to {100 * high:.1f}%"
        )
    return "\n".join(lines) + "\n"


def render_fresh(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    lines = [f"{'case':<8}{'package':<14}{'PyPI, 20 Sep':<14}answers said"]
    by_case: dict[str, list[Trace]] = {}
    for t in traces:
        by_case.setdefault(t.case_id, []).append(t)
    for case_id, case in cases.items():
        if case.slices["kind"] != "fresh":
            continue
        said = []
        for t in sorted(by_case.get(case_id, []), key=lambda t: t.trial):
            versions = sorted(set(VERSION.findall(t.answer)))
            said.append(", ".join(versions) if versions else "none")
        lines.append(
            f"{case_id:<8}{case.expected['relevant'][0]:<14}"
            f"{case.expected['version']:<14}{' | '.join(said)}"
        )
    return "\n".join(lines) + "\n"


def render_counterfactual(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    lines = [
        f"{'case':<8}{'package':<9}{'fact added':<12}{'retrieved':>10}{'followed it':>13}"
    ]
    by_case: dict[str, list[Trace]] = {}
    for t in traces:
        by_case.setdefault(t.case_id, []).append(t)
    for case_id, case in cases.items():
        if case.slices["kind"] != "counterfactual":
            continue
        runs = by_case.get(case_id, [])
        package = case.expected["relevant"][0]
        got = sum(any(r["name"] == package for r in t.retrieved) for t in runs)
        followed = sum(check(case, t, [])["ignored_context"] is False for t in runs)
        marker = case.expected["markers"][0]
        lines.append(
            f"{case_id:<8}{package:<9}{marker:<12}{_rate(got, len(runs)):>10}"
            f"{_rate(followed, len(runs)):>13}"
        )
    return "\n".join(lines) + "\n"


def flagged(
    cases: dict[str, EvalCase], traces: list[Trace], names: list[str]
) -> list[tuple[Trace, list[str]]]:
    """Every answer that one of the checks flags, with which ones."""
    out = []
    for trace in traces:
        found = check(cases[trace.case_id], trace, names)
        hits = [f for f in FLAGS if found[f]]
        if hits:
            out.append((trace, hits))
    return out
