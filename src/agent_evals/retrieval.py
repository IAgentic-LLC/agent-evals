"""Evaluate retrieval on its own, before any answer is written (chapter 12).

The package-intelligence product embeds each package's one-line summary, embeds the
question, and returns the three nearest summaries for the model to answer from. Every
answer starts there, so a package that is not in those three cannot be cited. This module
runs the product's own retrieval code with a recorded embedder and a scripted model, so
it needs no key and calls no language model, and scores what came back against a list of
packages that do the job asked.
"""

import asyncio
import json
import os
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from pkgintel_app.graph_ingestion import embed_and_upsert
from pkgintel_app.tenant_rag import ask_rag_agent_for_tenant
from qdrant_client import AsyncQdrantClient
from rank_bm25 import BM25Okapi
from reliable_agents_labs.models import ModelResult
from reliable_agents_labs.pypi import PackageMetadata

EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
# Words that carry no meaning for a keyword match. This is my list, kept short.
_WORDS = (
    "a an and the of to in for with on or my me your from is are it its that this by "
    "at as be do i we how what which something like"
)
STOPWORDS = frozenset(_WORDS.split())


def load_corpus(path: str | Path) -> list[dict[str, Any]]:
    lines = Path(path).read_text(encoding="utf8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------- metrics


def recall_at_k(ranked: list[str], relevant: list[str], k: int) -> float | None:
    """The share of the relevant packages found in the top k, or None with none relevant."""
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def hit_at_k(ranked: list[str], relevant: list[str], k: int) -> bool | None:
    """Is at least one relevant package in the top k?"""
    if not relevant:
        return None
    return bool(set(ranked[:k]) & set(relevant))


def precision_at_k(ranked: list[str], relevant: list[str], k: int) -> float | None:
    if not relevant:
        return None
    return len(set(ranked[:k]) & set(relevant)) / k


def reciprocal_rank(ranked: list[str], relevant: list[str]) -> float | None:
    """1 over the rank of the first relevant package, or 0 if none is ranked."""
    if not relevant:
        return None
    for rank, name in enumerate(ranked, 1):
        if name in relevant:
            return 1 / rank
    return 0.0


# ------------------------------------------------------------- embeddings


def vector_key(role: str, text: str) -> str:
    return f"{role}\t{text}"


def save_vectors(path: str | Path, keys: list[str], vectors: list[list[float]]) -> None:
    np.savez_compressed(
        path, keys=np.array(keys), vectors=np.array(vectors, dtype=np.float32)
    )


def load_vectors(path: str | Path) -> dict[str, list[float]]:
    data = np.load(path, allow_pickle=False)
    return {
        str(k): v.astype(float).tolist() for k, v in zip(data["keys"], data["vectors"])
    }


class RecordedEmbedder:
    """Answers `embed(text)` from a recording, for one role: `doc` or `query`."""

    def __init__(self, vectors: dict[str, list[float]], role: str) -> None:
        self._vectors, self._role = vectors, role

    async def embed(self, text: str) -> list[float]:
        return self._vectors[vector_key(self._role, text)]


class NativeEmbedder:
    """Gemini's own embedding endpoint, which takes a task type. The product's client
    goes through the OpenAI-compatible endpoint and passes none."""

    def __init__(self, task_type: str) -> None:
        self._task_type = task_type

    async def embed(self, text: str) -> list[float]:
        body = {
            "model": "models/gemini-embedding-001",
            "content": {"parts": [{"text": text}]},
            "taskType": self._task_type,
        }
        headers = {"x-goog-api-key": os.environ["GEMINI_API_KEY"]}
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(EMBED_URL, json=body, headers=headers)
            response.raise_for_status()
        return response.json()["embedding"]["values"]


async def record_vectors(
    corpus: list[dict[str, Any]],
    queries: list[str],
    doc_embed: Callable[[str], Any],
    query_embed: Callable[[str], Any],
    concurrency: int = 6,
) -> tuple[list[str], list[list[float]]]:
    """Embed every summary as a document and every question as a query, and keep them."""
    gate = asyncio.Semaphore(concurrency)
    items = [("doc", row["summary"]) for row in corpus] + [
        ("query", q) for q in queries
    ]

    async def one(role: str, text: str) -> list[float]:
        async with gate:
            return await (doc_embed if role == "doc" else query_embed)(text)

    vectors = await asyncio.gather(*(one(role, text) for role, text in items))
    return [vector_key(role, text) for role, text in items], list(vectors)


# --------------------------------------------------------------- rankers


class _NoModel:
    """A model that returns an empty answer, so the product's retrieval runs and no
    language model is called."""

    async def generate(self, *, system, user, tools=None, history=None):
        return ModelResult(
            text='{"answer": "", "cited_packages": []}',
            input_tokens=0,
            output_tokens=0,
            model_id="none",
            provider="none",
            tool_calls=[],
        )


async def ingest(
    qdrant: AsyncQdrantClient,
    tenant: str,
    rows: list[dict[str, Any]],
    doc_embedder: Any,
) -> None:
    """Put packages into a tenant's collection with the product's own ingestion step."""
    for row in rows:
        meta = PackageMetadata(
            name=row["name"], version=row["version"], summary=row["summary"]
        )
        await embed_and_upsert(meta, tenant, qdrant, doc_embedder)


async def product_rankings(
    corpus: list[dict[str, Any]],
    queries: dict[str, str],
    vectors: dict[str, list[float]],
    limit: int = 10,
    tenant: str = "acme",
) -> dict[str, list[dict[str, Any]]]:
    """What the product's retrieval returns for each question, best first."""
    qdrant = AsyncQdrantClient(":memory:")
    await ingest(qdrant, tenant, corpus, RecordedEmbedder(vectors, "doc"))
    found: dict[str, list[dict[str, Any]]] = {}
    for case_id, question in queries.items():
        await ask_rag_agent_for_tenant(
            tenant,
            question,
            qdrant=qdrant,
            embedder=RecordedEmbedder(vectors, "query"),
            model_client=_NoModel(),
            limit=limit,
            on_retrieval=lambda results, cid=case_id: found.__setitem__(cid, results),
        )
    return found


def tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def bm25_rankings(
    corpus: list[dict[str, Any]], queries: dict[str, str], limit: int = 10
) -> dict[str, list[dict[str, Any]]]:
    """A keyword baseline over the same text the product embeds: the summary."""
    index = BM25Okapi([tokens(row["summary"]) for row in corpus])
    found = {}
    for case_id, question in queries.items():
        scores = index.get_scores(tokens(question))
        order = sorted(
            range(len(corpus)), key=lambda i: (-scores[i], corpus[i]["name"])
        )
        found[case_id] = [
            {"name": corpus[i]["name"], "score": float(scores[i])}
            for i in order[:limit]
        ]
    return found


def rrf(rankings: list[list[str]], k: int = 60, limit: int = 10) -> list[str]:
    """Reciprocal rank fusion: each list gives a package 1 / (k + rank), and the sums win."""
    score: dict[str, float] = {}
    for ranked in rankings:
        for rank, name in enumerate(ranked, 1):
            score[name] = score.get(name, 0.0) + 1 / (k + rank)
    return sorted(score, key=lambda n: (-score[n], n))[:limit]


# ----------------------------------------------------------- tenants


async def leaked_results(
    corpus: list[dict[str, Any]],
    queries: dict[str, str],
    vectors: dict[str, list[float]],
    shared: bool,
    limit: int = 3,
) -> tuple[int, int]:
    """Two tenants each own half the packages. Ask as the first tenant and count the
    results that belong to the second. `shared` puts both tenants in one collection,
    the fault: nothing then separates them."""
    mine, theirs = corpus[0::2], corpus[1::2]
    qdrant = AsyncQdrantClient(":memory:")
    docs = RecordedEmbedder(vectors, "doc")
    first, second = ("shared", "shared") if shared else ("acme", "globex")
    await ingest(qdrant, first, mine, docs)
    await ingest(qdrant, second, theirs, docs)
    allowed = {row["name"] for row in mine}
    leaked = total = 0
    for question in queries.values():
        captured: list[dict[str, Any]] = []
        await ask_rag_agent_for_tenant(
            first,
            question,
            qdrant=qdrant,
            embedder=RecordedEmbedder(vectors, "query"),
            model_client=_NoModel(),
            limit=limit,
            on_retrieval=captured.extend,
        )
        total += len(captured)
        leaked += sum(item["name"] not in allowed for item in captured)
    return leaked, total


def bootstrap_interval(
    values: list[float], resamples: int = 10_000, seed: int = 0
) -> tuple[float, float]:
    """A 95% interval for the mean, from resampling the queries with replacement."""
    rng = np.random.default_rng(seed)
    data = np.array(values)
    means = rng.choice(data, size=(resamples, len(data)), replace=True).mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def auc(positive: list[float], negative: list[float]) -> float:
    """The chance a random answerable question scores higher than a random unanswerable one."""
    wins = sum((p > n) + 0.5 * (p == n) for p in positive for n in negative)
    return wins / (len(positive) * len(negative))
