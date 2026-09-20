"""Checks on an evaluation dataset: is it well formed, is it balanced, does it leak.

Leakage here means a test case that is too close to something the system was
built against. Two measures look for it: word overlap, which catches copies and
light edits, and embedding similarity, which also catches paraphrases. Each
similarity score means little alone, so a few known paraphrases are scored the
same way as a yardstick.
"""

import math
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from agent_evals.schema import EvalCase

HANDLERS = {"billing", "technical", "security"}
MIN_SLICE = 10  # below this, a slice's interval is too wide to say much


@dataclass
class Issue:
    severity: str  # "error" or "warning"
    where: str
    message: str


@dataclass
class Neighbor:
    case_id: str
    reference_id: str
    score: float


def check_cases(cases: list[EvalCase]) -> list[Issue]:
    issues: list[Issue] = []
    seen: Counter[str] = Counter(c.case_id for c in cases)
    for case_id, n in seen.items():
        if n > 1:
            issues.append(Issue("error", case_id, f"case_id appears {n} times"))
    slice_keys = [set(c.slices) for c in cases]
    for c in cases:
        for field in ("subject", "body"):
            if not str(c.input.get(field, "")).strip():
                issues.append(Issue("error", c.case_id, f"input.{field} is empty"))
        if c.input.get("category") not in HANDLERS:
            issues.append(Issue("error", c.case_id, "input.category is not valid"))
        if c.expected.get("handled_by") not in HANDLERS:
            issues.append(Issue("error", c.case_id, "expected.handled_by is not valid"))
        if not c.expected.get("required_actions"):
            issues.append(
                Issue("error", c.case_id, "expected.required_actions is empty")
            )
        forbids = bool(c.invariants.get("forbidden_actions"))
        if (c.slices.get("adversarial") == "yes") != forbids:
            issues.append(
                Issue(
                    "error",
                    c.case_id,
                    "adversarial slice and forbidden_actions disagree",
                )
            )
        if not c.split:
            issues.append(Issue("error", c.case_id, "split is empty"))
        if c.split == "test" and not c.canary:
            issues.append(Issue("error", c.case_id, "test case has no canary"))
    if slice_keys and any(keys != slice_keys[0] for keys in slice_keys):
        issues.append(
            Issue("error", "dataset", "cases do not all use the same slice keys")
        )
    for key, values in slice_counts(cases).items():
        for value, n in values.items():
            if n < MIN_SLICE:
                issues.append(
                    Issue(
                        "warning",
                        f"{key}={value}",
                        f"only {n} case(s), under {MIN_SLICE}",
                    )
                )
    return issues


def slice_counts(cases: list[EvalCase]) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = {}
    for c in cases:
        for key, value in c.slices.items():
            counts.setdefault(key, Counter())[value] += 1
    return {key: dict(sorted(values.items())) for key, values in sorted(counts.items())}


def ticket_text(case: EvalCase) -> str:
    return f"{case.input.get('subject', '')}. {case.input.get('body', '')}"


def _shingles(text: str, n: int = 3) -> set[tuple[str, ...]]:
    words = re.findall(r"\w+", text.lower())
    return {tuple(words[i : i + n]) for i in range(max(1, len(words) - n + 1))}


def jaccard(a: str, b: str) -> float:
    sa, sb = _shingles(a), _shingles(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


def nearest_lexical(cases: list[EvalCase], reference: list[EvalCase]) -> list[Neighbor]:
    """For each case, its most word-similar case in the reference set."""
    out = []
    for c in cases:
        best = max(reference, key=lambda r: jaccard(ticket_text(c), ticket_text(r)))
        out.append(
            Neighbor(
                c.case_id, best.case_id, jaccard(ticket_text(c), ticket_text(best))
            )
        )
    return out


def cosine(u: list[float], v: list[float]) -> float:
    dot = sum(a * b for a, b in zip(u, v, strict=True))
    norm = math.sqrt(sum(a * a for a in u)) * math.sqrt(sum(b * b for b in v))
    return dot / norm if norm else 0.0


Embed = Callable[[str], Awaitable[list[float]]]


async def nearest_semantic(
    cases: list[EvalCase], reference: list[EvalCase], embed: Embed
) -> list[Neighbor]:
    """For each case, its most similar case in the reference set by embedding."""
    ref_vectors = {r.case_id: await embed(ticket_text(r)) for r in reference}
    out = []
    for c in cases:
        vector = await embed(ticket_text(c))
        best_id, best = max(
            ((rid, cosine(vector, rv)) for rid, rv in ref_vectors.items()),
            key=lambda pair: pair[1],
        )
        out.append(Neighbor(c.case_id, best_id, best))
    return out


async def control_scores(
    controls: list[tuple[str, str]], reference: list[EvalCase], embed: Embed
) -> list[Neighbor]:
    """Similarity of each known paraphrase to the case it paraphrases: the yardstick."""
    by_id = {r.case_id: r for r in reference}
    out = []
    for source_id, text in controls:
        score = cosine(await embed(text), await embed(ticket_text(by_id[source_id])))
        out.append(Neighbor(f"paraphrase of {source_id}", source_id, score))
    return out


def near_dev_flags(
    semantic: list[Neighbor], controls: list[Neighbor], embedding_model: str
) -> dict:
    """A derived slice: which cases are about as close to a development case as a
    deliberate paraphrase is. Written beside the dataset so the dataset stays frozen.
    """
    threshold = min(c.score for c in controls)
    return {
        "key": "near_dev",
        "meaning": (
            "yes when the ticket is at least as similar to a development ticket as "
            "the least similar known paraphrase"
        ),
        "embedding_model": embedding_model,
        "threshold": round(threshold, 4),
        "values": {
            n.case_id: "yes" if n.score >= threshold else "no" for n in semantic
        },
        "nearest": {
            n.case_id: {
                "development_case": n.reference_id,
                "similarity": round(n.score, 4),
            }
            for n in semantic
        },
    }


def render_check(
    name: str,
    cases: list[EvalCase],
    issues: list[Issue],
    lexical: list[Neighbor] | None = None,
    semantic: list[Neighbor] | None = None,
    controls: list[Neighbor] | None = None,
    top: int = 5,
) -> str:
    lines = [f"Dataset check: {name} ({len(cases)} cases)", ""]
    errors = [i for i in issues if i.severity == "error"]
    warnings = [i for i in issues if i.severity == "warning"]
    lines.append(f"Structure: {len(errors)} error(s), {len(warnings)} warning(s)")
    lines += [f"  {i.severity.upper():<7} {i.where}: {i.message}" for i in issues]
    lines += ["", "Slices (cases per value):"]
    for key, values in slice_counts(cases).items():
        lines.append(f"  {key:<17}" + "  ".join(f"{v} {n}" for v, n in values.items()))

    def block(title: str, rows: list[Neighbor]) -> None:
        lines.extend(["", title])
        for r in sorted(rows, key=lambda r: -r.score)[:top]:
            lines.append(f"  {r.score:.2f}  {r.case_id} is closest to {r.reference_id}")

    if lexical is not None:
        block(
            "Word overlap with the reference set (3-word phrases, 1.00 = identical):",
            lexical,
        )
    if controls:
        lo, hi = min(c.score for c in controls), max(c.score for c in controls)
        lines += [
            "",
            f"Known paraphrases score {lo:.2f} to {hi:.2f} against their sources.",
        ]
    if semantic is not None:
        block("Embedding similarity to the reference set (cosine):", semantic)
        if controls:
            lo = min(c.score for c in controls)
            over = [s for s in semantic if s.score >= lo]
            lines.append(
                f"  {len(over)} of {len(semantic)} cases score at or above the lowest "
                f"known paraphrase ({lo:.2f})."
            )
    return "\n".join(lines) + "\n"
