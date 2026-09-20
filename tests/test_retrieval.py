"""Chapter 12: retrieval scored on its own, before any answer is written.

No key needed: hand-made vectors, and the recorded embeddings under runs/pkg-retrieval-*.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from agent_evals import cli
from agent_evals import retrieval as R
from agent_evals import retrieval_report as RR
from agent_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datasets/pkg_corpus_v1.jsonl"
V1 = ROOT / "datasets/pkg_queries_v1.jsonl"
V2 = ROOT / "datasets/pkg_queries_v2.jsonl"
RUN = ROOT / "runs/pkg-retrieval-2"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ---------------------------------------------------------------- metrics


def test_the_metrics_on_the_worked_examples():
    charts = ["seaborn", "pandas", "matplotlib", "ImageIO", "plotly"]
    relevant = ["altair", "bokeh", "matplotlib", "plotly", "seaborn"]
    assert R.hit_at_k(charts, relevant, 3) is True
    assert R.recall_at_k(charts, relevant, 3) == pytest.approx(0.4)
    assert R.precision_at_k(charts, relevant, 3) == pytest.approx(2 / 3)
    assert R.reciprocal_rank(charts, relevant) == 1.0
    tox = ["coverage", "tox", "pytest-cov"]
    assert R.reciprocal_rank(tox, ["tox"]) == 0.5
    assert R.reciprocal_rank(["a", "b"], ["z"]) == 0.0


def test_a_question_with_no_relevant_package_has_no_score():
    assert R.hit_at_k(["a"], [], 3) is None
    assert R.recall_at_k(["a"], [], 3) is None
    assert R.reciprocal_rank(["a"], []) is None


def test_reciprocal_rank_fusion_rewards_agreement_and_is_deterministic():
    fused = R.rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert fused[:2] == ["a", "b"] and set(fused) == {"a", "b", "c", "d"}
    assert R.rrf([["a", "b"], ["b", "a"]]) == R.rrf([["a", "b"], ["b", "a"]])


def test_the_bootstrap_interval_is_reproducible_and_brackets_the_mean():
    values = [0.0, 0.5, 1.0, 1.0, 0.25, 0.75, 1.0, 0.0]
    low, high = R.bootstrap_interval(values)
    assert (low, high) == R.bootstrap_interval(values)
    assert low < R.mean(values) < high


def test_auc_is_one_when_every_positive_outscores_every_negative():
    assert R.auc([0.9, 0.8], [0.5, 0.4]) == 1.0
    assert R.auc([0.5], [0.5]) == 0.5


# ------------------------------------------------------------ the product path


def _one_hot(i):
    vector = [0.0] * 3072
    vector[i] = 1.0
    return vector


def _tiny():
    corpus = [
        {"name": "alpha", "version": "1", "summary": "first thing"},
        {"name": "beta", "version": "1", "summary": "second thing"},
        {"name": "gamma", "version": "1", "summary": "third thing"},
    ]
    vectors = {
        R.vector_key("doc", r["summary"]): _one_hot(i) for i, r in enumerate(corpus)
    }
    vectors[R.vector_key("query", "want the second")] = _one_hot(1)
    return corpus, vectors


async def test_the_products_own_retrieval_returns_the_nearest_summary_first():
    corpus, vectors = _tiny()
    found = await R.product_rankings(corpus, {"Q": "want the second"}, vectors, limit=3)
    assert found["Q"][0]["name"] == "beta"
    assert found["Q"][0]["score"] > found["Q"][1]["score"]


async def test_a_shared_collection_leaks_and_one_collection_per_tenant_does_not():
    corpus, vectors = _tiny()
    corpus = corpus * 1
    # The first tenant owns alpha and gamma. The second owns beta.
    queries = {"Q": "want the second"}
    leaked, total = await R.leaked_results(
        corpus, queries, vectors, shared=False, limit=3
    )
    assert (leaked, total) == (0, 2)  # only alpha and gamma exist in its collection
    leaked, total = await R.leaked_results(
        corpus, queries, vectors, shared=True, limit=3
    )
    assert leaked == 1 and total == 3  # beta belongs to the other tenant


async def test_ingestion_uses_a_collection_per_tenant():
    corpus, vectors = _tiny()
    qdrant = AsyncQdrantClient(":memory:")
    await R.ingest(qdrant, "acme", corpus[:1], R.RecordedEmbedder(vectors, "doc"))
    await R.ingest(qdrant, "globex", corpus[1:], R.RecordedEmbedder(vectors, "doc"))
    names = {c.name for c in (await qdrant.get_collections()).collections}
    assert names == {"packages_acme", "packages_globex"}


def test_bm25_finds_a_package_by_a_word_in_its_summary():
    corpus, _ = _tiny()
    found = R.bm25_rankings(corpus, {"Q": "the second one"}, limit=3)
    assert found["Q"][0]["name"] == "beta"


# ----------------------------------------------------------- the datasets


def test_the_corpus_is_107_unique_packages_with_summaries_and_a_date():
    rows = R.load_corpus(CORPUS)
    assert len(rows) == 107 and len({r["name"] for r in rows}) == 107
    assert all(r["summary"].strip() and r["fetched_at"] == "2026-09-20" for r in rows)


def test_every_relevant_package_exists_and_the_versions_are_pinned():
    names = {r["name"] for r in R.load_corpus(CORPUS)}
    v1, v2 = load_cases(V1), load_cases(V2)
    assert (len(v1), len(v2)) == (50, 62)
    for case in v2:
        assert set(case.expected["relevant"]) <= names and case.canary
    assert [c.model_dump() for c in v2[:50]] == [c.model_dump() for c in v1]
    assert _sha(V1)[:8] == "7cb9eeef" and _sha(V2)[:8] == "38f5b804"
    kinds = {}
    for c in v2:
        kinds[c.slices["kind"]] = kinds.get(c.slices["kind"], 0) + 1
    assert kinds == {"task": 40, "named": 5, "none": 5, "hard": 12}
    assert sum(c.slices["label_confidence"] == "contested" for c in v2) == 16


def test_the_recordings_match_the_dataset_files_they_were_made_from():
    manifest = json.loads((RUN / "manifest.json").read_text(encoding="utf8"))
    assert manifest["corpus"]["sha256"] == _sha(CORPUS)
    assert manifest["queries"]["sha256"] == _sha(V2)
    assert manifest["harness"]["dirty"] is False
    old = json.loads(
        (ROOT / "runs/pkg-retrieval-1/manifest.json").read_text(encoding="utf8")
    )
    assert old["queries"]["sha256"] == _sha(V1)


# ------------------------------------------------------- the recorded results


@pytest.fixture(scope="module")
def study():
    s = RR.Study(
        R.load_corpus(CORPUS),
        load_cases(V2),
        R.load_vectors(RUN / "embeddings-shipped.npz"),
        R.load_vectors(RUN / "embeddings-typed.npz"),
    )
    return asyncio.run(s.run())


def test_hit_at_3_by_method_on_57_answerable_questions(study):
    assert len(study.answerable) == 57
    got = {m: sum(study.hits(m, 3)) for m in RR.METHODS}
    assert got == {
        "random (seed 0)": 5,
        "alphabetical": 2,
        "BM25 keywords": 39,
        "hybrid (dense + BM25)": 48,
        "dense, as shipped": 55,
        "dense, task types": 56,
    }


def test_recall_and_rank_for_the_two_dense_versions(study):
    shipped, typed = RR.SHIPPED, RR.TYPED
    assert round(R.mean(study.recalls(shipped, 3)), 2) == 0.75
    assert round(R.mean(study.recalls(typed, 3)), 2) == 0.83
    assert round(R.mean(study.reciprocals(shipped)), 2) == 0.90
    assert round(R.mean(study.reciprocals(typed)), 2) == 0.96
    assert sum(study.hits(shipped, 5)) == 57  # five results always include one


def test_the_only_misses_are_the_two_contested_questions(study):
    misses = [q for q in study.answerable if not study.hits(RR.SHIPPED, 3, [q])[0]]
    assert misses == ["PQ-042", "PQ-058"]
    assert not any(study.clear[q] for q in misses)
    assert study.rank[RR.SHIPPED]["PQ-058"][:3] == ["python-docx", "click", "fire"]
    clear = [q for q in study.answerable if study.clear[q]]
    assert len(clear) == 41
    assert all(study.hits(RR.SHIPPED, 3, clear)) and all(study.hits(RR.TYPED, 3, clear))


def test_task_types_beat_the_shipped_embedder_on_eleven_questions_and_lose_on_none(
    study,
):
    diff = [
        a - b for a, b in zip(study.recalls(RR.TYPED, 3), study.recalls(RR.SHIPPED, 3))
    ]
    assert (sum(d > 0 for d in diff), sum(d < 0 for d in diff)) == (11, 0)
    low, _ = R.bootstrap_interval(diff)
    assert low > 0 and round(R.mean(diff), 3) == 0.085


def test_the_scores_of_questions_with_no_answer_overlap_the_hardest_answerable_one(
    study,
):
    top = study.top1["shipped"]
    none = [top[q] for q in study.queries if not study.relevant[q]]
    lowest = min(top[q] for q in study.answerable)
    assert len(none) == 5 and lowest < max(none)  # 0.575 against 0.594


def test_isolation_holds_with_a_collection_per_tenant_and_fails_when_shared(study):
    vectors = R.load_vectors(RUN / "embeddings-shipped.npz")
    kept = asyncio.run(
        R.leaked_results(study.corpus, study.queries, vectors, shared=False)
    )
    shared = asyncio.run(
        R.leaked_results(study.corpus, study.queries, vectors, shared=True)
    )
    assert kept == (0, 186) and shared == (103, 186)


def test_the_report_commands_exit_0_and_the_isolation_command_checks_both_cases(capsys):
    base = ["--run", str(RUN), "--queries", str(V2), "--corpus", str(CORPUS)]
    for part in ("methods", "kinds", "paired", "misses", "scores"):
        assert cli.main(["retrieval", "report", *base, "--part", part]) == 0
    out = capsys.readouterr().out
    assert "dense, as shipped" in out and "PQ-058" in out
    assert cli.main(["retrieval", "isolation", *base]) == 0
    assert "103 of 186" in capsys.readouterr().out
