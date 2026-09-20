"""When the right answer is "the package information does not say" (chapter 14).

For each question there is a right behavior given what was retrieved. If a relevant
package was among the three, the product should answer. If not, or if the question asks
for something no one-line summary holds, it should decline. This module labels that
from the dataset and the trace, counts what the product did, and asks how well the top
similarity score could stand in for the model's own decision.
"""

import re
from collections import Counter
from statistics import median

from agent_evals.retrieval import STOPWORDS, hit_at_k, tokens
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

ABSTAIN_KINDS = ("none", "fresh", "outside", "beyond_summary", "false_premise")
CONFIGS = {
    "shipped prompt": ("pkg-abstain-1", "pkg-abstain-2"),
    "permissive prompt": ("pkg-abstain-perm-1", "pkg-abstain-perm-2"),
}


def should_answer(case: EvalCase, trace: Trace) -> bool:
    """True when a relevant package was retrieved for a question the index can answer."""
    if case.slices["kind"] in ABSTAIN_KINDS:
        return False
    names = [r["name"] for r in trace.retrieved]
    return bool(hit_at_k(names, case.expected["relevant"], len(names)))


def group(case: EvalCase, trace: Trace) -> str:
    """The row a question belongs to in the tables: its kind, or `retrieval miss`."""
    kind = case.slices["kind"]
    if kind in ABSTAIN_KINDS:
        return kind
    return "should answer" if should_answer(case, trace) else "retrieval miss"


def answered(trace: Trace) -> bool:
    """The product's own signal: an answer that cites something. Declining leaves
    `cited_packages` empty, as the prompt says."""
    return trace.error is None and bool(trace.cited)


def _interval(k: int, n: int) -> str:
    low, high = wilson_interval(k, n)
    return f"{100 * low:.0f}-{100 * high:.0f}%"


def _pairs(cases, runs):
    return [(cases[t.case_id], t) for traces in runs for t in traces]


def render_matrix(
    cases: dict[str, EvalCase], runs: dict[str, list[list[Trace]]]
) -> str:
    head = f"{'':<19}{'refused, should answer':>26}{'answered, should not':>26}"
    lines = [head]
    for name, traces in runs.items():
        over = n_yes = unsafe = n_no = 0
        for case, t in _pairs(cases, traces):
            if should_answer(case, t):
                n_yes += 1
                over += not answered(t)
            else:
                n_no += 1
                unsafe += answered(t)
        lines.append(
            f"{name:<19}{f'{over} of {n_yes}':>11}{_interval(over, n_yes):>9}"
            f"{f'{unsafe} of {n_no}':>15}{_interval(unsafe, n_no):>11}"
        )
    return "\n".join(lines) + "\n"


def render_kinds(cases: dict[str, EvalCase], runs: dict[str, list[list[Trace]]]) -> str:
    rows = ("should answer", *ABSTAIN_KINDS[2:], "none", "fresh", "retrieval miss")
    rows = tuple(dict.fromkeys(rows))
    names = list(runs)
    lines = [
        f"{'answers that cite something':<29}" + "".join(f"{n:>19}" for n in names)
    ]
    for row in rows:
        cells = []
        for name in names:
            k = n = 0
            for case, t in _pairs(cases, runs[name]):
                if group(case, t) == row:
                    n += 1
                    k += answered(t)
            cells.append(f"{k} of {n}")
        lines.append(f"{row:<29}" + "".join(f"{c:>19}" for c in cells))
    return "\n".join(lines) + "\n"


def render_paired(
    cases: dict[str, EvalCase], runs: dict[str, list[list[Trace]]]
) -> str:
    """Trial by trial, on the questions that should be answered: who refused."""
    (_, first), (_, second) = runs.items()
    both = only_a = only_b = neither = 0
    for one, two in zip(first, second):
        for a, b in zip(one, two):
            if not should_answer(cases[a.case_id], a):
                continue
            ra, rb = not answered(a), not answered(b)
            both += ra and rb
            only_a += ra and not rb
            only_b += rb and not ra
            neither += not ra and not rb
    names = list(runs)
    return (
        f"Trial by trial, on the questions that should be answered\n"
        f"  refused under both prompts      {both:>4}\n"
        f"  refused under {names[0] + ' only':<17}{only_a:>4}\n"
        f"  refused under {names[1] + ' only':<17}{only_b:>4}\n"
        f"  answered under both             {neither:>4}\n"
    )


def _top1(t: Trace) -> float:
    return float(t.retrieved[0]["score"])


def render_scores(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    rows: dict[str, list[float]] = {}
    for case, t in _pairs(cases, [traces]):
        rows.setdefault(group(case, t), []).append(_top1(t))
    order = ("should answer", "outside", "beyond_summary", "false_premise", "none")
    order += ("fresh", "retrieval miss")
    lines = [
        f"{'top-1 cosine similarity':<26}{'n':>4}{'lowest':>9}{'median':>9}{'highest':>9}"
    ]
    for row in order:
        v = sorted(rows[row])
        lines.append(f"{row:<26}{len(v):>4}{v[0]:>9.3f}{median(v):>9.3f}{v[-1]:>9.3f}")
    return "\n".join(lines) + "\n"


def _gate(cases, traces, split, cutoff):
    """Questions a cutoff on the top score would refuse before any model is called."""
    caught = wrong = should_not = should = 0
    for case, t in _pairs(cases, [traces]):
        if case.split != split:
            continue
        low = _top1(t) < cutoff
        if should_answer(case, t):
            should += 1
            wrong += low
        else:
            should_not += 1
            caught += low
    return caught, should_not, wrong, should


def choose_cutoff(cases: dict[str, EvalCase], traces: list[Trace]) -> float:
    """The cutoff on the dev questions that refuses the most questions that should be
    declined minus the most that should be answered, counting each the same. That
    weighting is mine and is a product decision. Ties go to the lower cutoff."""
    best, best_score = 0.0, None
    for step in range(500, 800):
        cutoff = step / 1000
        caught, _, wrong, _ = _gate(cases, traces, "dev", cutoff)
        if best_score is None or caught - wrong > best_score:
            best, best_score = cutoff, caught - wrong
    return best


def render_gate(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    chosen = choose_cutoff(cases, traces)
    head = (
        f"{'cutoff':<8}{'dev: caught':>14}{'wrongly refused':>17}"
        f"{'test: caught':>15}{'wrongly refused':>17}"
    )
    lines = [head]
    for cutoff in sorted({0.58, 0.60, 0.62, 0.64, 0.66, 0.68, chosen}):
        dc, dn, dw, dy = _gate(cases, traces, "dev", cutoff)
        tc, tn, tw, ty = _gate(cases, traces, "test", cutoff)
        mark = " <-" if cutoff == chosen else ""
        lines.append(
            f"{cutoff:<8.3f}{f'{dc} of {dn}':>14}{f'{dw} of {dy}':>17}"
            f"{f'{tc} of {tn}':>15}{f'{tw} of {ty}':>17}{mark}"
        )
    return "\n".join(lines) + "\n"


_WORDS = STOPWORDS | {"can", "does", "do", "use", "using", "make", "made", "way"}


def _overlap(case: EvalCase, trace: Trace) -> float:
    """The share of the question's words found in the relevant retrieved summaries."""
    question = {w for w in tokens(case.input["query"]) if w not in _WORDS}
    relevant = set(case.expected["relevant"])
    text = " ".join(
        f"{r['name']} {r['summary']}" for r in trace.retrieved if r["name"] in relevant
    )
    have = set(re.findall(r"[a-z0-9]+", text.lower()))
    return len(question & have) / len(question) if question else 0.0


def render_overlap(
    cases: dict[str, EvalCase], runs: dict[str, list[list[Trace]]]
) -> str:
    """Do refusals go with summaries that share no word with the question?"""
    lines = [f"{'':<19}{'shares a word':>15}{'shares no word':>16}"]
    for name, traces in runs.items():
        cell = {True: Counter(), False: Counter()}
        for case, t in _pairs(cases, traces):
            if not should_answer(case, t):
                continue
            shares = _overlap(case, t) > 0
            cell[shares]["n"] += 1
            cell[shares]["refused"] += not answered(t)
        yes, no = cell[True], cell[False]
        a = f"{yes['refused']} of {yes['n']}"
        b = f"{no['refused']} of {no['n']}"
        lines.append(f"{name:<19}{a:>15}{b:>16}")
    return "\n".join(lines) + "\n"
