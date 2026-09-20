"""Reports on retrieval, read from recorded embeddings (chapter 12)."""

import random
from collections.abc import Callable
from typing import Any

from agent_evals.retrieval import (
    bm25_rankings,
    bootstrap_interval,
    hit_at_k,
    mean,
    product_rankings,
    recall_at_k,
    reciprocal_rank,
    rrf,
)
from agent_evals.stats import wilson_interval

METHODS = (
    "random (seed 0)",
    "alphabetical",
    "BM25 keywords",
    "hybrid (dense + BM25)",
    "dense, as shipped",
    "dense, task types",
)
SHIPPED, TYPED = "dense, as shipped", "dense, task types"


class Study:
    """Every method's ranking for every question, so each report reads the same data."""

    def __init__(
        self,
        corpus: list[dict[str, Any]],
        cases: list[Any],
        shipped: dict[str, list[float]],
        typed: dict[str, list[float]],
    ) -> None:
        self.corpus = corpus
        self.queries = {c.case_id: c.input["query"] for c in cases}
        self.relevant = {c.case_id: c.expected["relevant"] for c in cases}
        self.kind = {c.case_id: c.slices["kind"] for c in cases}
        self.clear = {c.case_id: c.slices["label_confidence"] == "clear" for c in cases}
        self.answerable = [q for q in self.queries if self.relevant[q]]
        self._vectors = {"shipped": shipped, "typed": typed}
        self.rank: dict[str, dict[str, list[str]]] = {}
        self.top1: dict[str, dict[str, float]] = {}

    async def run(self) -> "Study":
        found = {}
        for kind, vectors in self._vectors.items():
            found[kind] = await product_rankings(self.corpus, self.queries, vectors)
        keyword = bm25_rankings(self.corpus, self.queries)

        def names(d: dict[str, list[dict[str, Any]]]) -> dict[str, list[str]]:
            return {q: [x["name"] for x in items] for q, items in d.items()}

        shipped, typed = names(found["shipped"]), names(found["typed"])
        bm25 = names(keyword)
        every = [row["name"] for row in self.corpus]
        rng = random.Random(0)
        self.rank = {
            "random (seed 0)": {q: rng.sample(every, 10) for q in self.queries},
            "alphabetical": {q: sorted(every)[:10] for q in self.queries},
            "BM25 keywords": bm25,
            "hybrid (dense + BM25)": {
                q: rrf([shipped[q], bm25[q]]) for q in self.queries
            },
            SHIPPED: shipped,
            TYPED: typed,
        }
        self.top1 = {
            kind: {q: items[0]["score"] for q, items in found[kind].items()}
            for kind in found
        }
        return self

    def _each(self, method: str, metric: Callable[..., Any], queries: list[str] | None):
        return [
            metric(self.rank[method][q], self.relevant[q])
            for q in (queries if queries is not None else self.answerable)
        ]

    def hits(self, method: str, k: int, queries: list[str] | None = None) -> list[bool]:
        return self._each(method, lambda r, rel: hit_at_k(r, rel, k), queries)

    def recalls(
        self, method: str, k: int, queries: list[str] | None = None
    ) -> list[float]:
        return self._each(method, lambda r, rel: recall_at_k(r, rel, k), queries)

    def reciprocals(self, method: str, queries: list[str] | None = None) -> list[float]:
        return self._each(method, reciprocal_rank, queries)


def _interval(k: int, n: int) -> str:
    low, high = wilson_interval(k, n)
    return f"{100 * low:.0f}-{100 * high:.0f}%"


def render_methods(study: Study, run: str) -> str:
    n = len(study.answerable)
    head = f"{'method':<22}{'hit@1':>6}{'hit@3':>7}{'hit@5':>7}{'rec@3':>7}{'MRR':>6}"
    lines = [
        f"Retrieval on {n} answerable questions, {len(study.corpus)} packages ({run})",
        "",
        head + "   hit@3 interval",
    ]
    for method in METHODS:
        h1, h3, h5 = (sum(study.hits(method, k)) for k in (1, 3, 5))
        rec, mrr = mean(study.recalls(method, 3)), mean(study.reciprocals(method))
        lines.append(
            f"{method:<22}{h1 / n:>6.2f}{h3 / n:>7.2f}{h5 / n:>7.2f}{rec:>7.2f}{mrr:>6.2f}"
            f"   {_interval(h3, n)}"
        )
    sizes = [len(study.relevant[q]) for q in study.answerable]
    ceiling = mean([min(3, s) / s for s in sizes])
    clear = [q for q in study.answerable if study.clear[q]]
    lines += [
        "",
        f"Best possible rec@3, given how many packages are relevant: {ceiling:.2f}",
        (
            f"On the {len(clear)} questions with clear labels, dense as shipped has "
            f"hit@3 {sum(study.hits(SHIPPED, 3, clear))} of {len(clear)}."
        ),
    ]
    return "\n".join(lines) + "\n"


def render_kinds(study: Study) -> str:
    kinds = ["task", "named", "hard"]
    groups = {k: [q for q in study.answerable if study.kind[q] == k] for k in kinds}
    head = f"{'hit@3 and rec@3':<22}"
    head += "".join(f"{f'{k} ({len(groups[k])})':>15}" for k in kinds)
    lines = [head]
    for method in METHODS[2:]:
        cells = []
        for k in kinds:
            qs = groups[k]
            hit, rec = (
                sum(study.hits(method, 3, qs)),
                mean(study.recalls(method, 3, qs)),
            )
            cells.append(f"{hit}/{len(qs)}  {rec:.2f}")
        lines.append(f"{method:<22}" + "".join(f"{c:>15}" for c in cells))
    return "\n".join(lines) + "\n"


def render_paired(study: Study) -> str:
    rows = [
        ("types vs shipped", TYPED, SHIPPED, "rec@3"),
        ("types vs shipped", TYPED, SHIPPED, "MRR"),
        ("shipped vs BM25", SHIPPED, "BM25 keywords", "rec@3"),
        ("shipped vs hybrid", SHIPPED, "hybrid (dense + BM25)", "rec@3"),
    ]
    head = (
        f"{'first minus second':<19}{'metric':<6}{'better':>7}{'worse':>6}{'same':>5}"
    )
    lines = [head + "   mean diff (95% interval)"]
    for label, a, b, metric in rows:
        get = (
            (lambda m: study.recalls(m, 3)) if metric == "rec@3" else study.reciprocals
        )
        diff = [x - y for x, y in zip(get(a), get(b))]
        low, high = bootstrap_interval(diff)
        better, worse = sum(d > 0 for d in diff), sum(d < 0 for d in diff)
        lines.append(
            f"{label:<19}{metric:<6}{better:>7}{worse:>6}{len(diff) - better - worse:>5}"
            f"   {mean(diff):+.3f} ({low:+.3f} to {high:+.3f})"
        )
    return "\n".join(lines) + "\n"


def render_misses(study: Study, method: str = SHIPPED) -> str:
    lines = [f"Questions where {method} has no relevant package in its top 3"]
    for q in study.answerable:
        ranked = study.rank[method][q]
        if hit_at_k(ranked, study.relevant[q], 3):
            continue
        first = next(
            (i for i, n in enumerate(ranked, 1) if n in study.relevant[q]), None
        )
        label = "clear" if study.clear[q] else "contested"
        lines += [
            f'{q} ({study.kind[q]}, {label}): "{study.queries[q]}"',
            f"  returned: {', '.join(ranked[:3])}",
            f"  relevant: {', '.join(study.relevant[q])} (first at rank {first})",
        ]
    return "\n".join(lines) + "\n"


def render_scores(study: Study) -> str:
    lines = [f"{'top-1 cosine similarity':<32}{'lowest':>8}{'median':>8}{'highest':>9}"]
    for kind, label in (("shipped", "as shipped"), ("typed", "task types")):
        top = study.top1[kind]
        yes = sorted(top[q] for q in study.answerable)
        no = sorted(top[q] for q in study.queries if not study.relevant[q])
        for name, values in (
            (f"{label}, answerable", yes),
            (f"{label}, no answer", no),
        ):
            mid = values[len(values) // 2]
            lines.append(f"{name:<32}{values[0]:>8.3f}{mid:>8.3f}{values[-1]:>9.3f}")
    return "\n".join(lines) + "\n"
