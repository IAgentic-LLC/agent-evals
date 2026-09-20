"""Drive the package-intelligence product's answer path (chapter 13).

The product's own `ask_rag_agent_for_tenant` runs unchanged: it embeds the question,
searches the tenant's collection, puts the three nearest summaries in the prompt and
asks the model for a cited answer. Only two things are replaced. Embeddings come from
the recording made in chapter 12's way, so retrieval is the same on every run. And the
model is the real one, or a scripted one for the checks that need no key.

A case may carry `summary_edits`: package names mapped to a new summary. Those
questions are asked against a copy of the index with the edits applied.
"""

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

from pkgintel_app.tenant_rag import ask_rag_agent_for_tenant
from qdrant_client import AsyncQdrantClient
from reliable_agents_labs.models import ModelResult

from agent_evals.retrieval import RecordedEmbedder, ingest, load_corpus, load_vectors
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "datasets" / "pkg_corpus_v1.jsonl"
VECTORS = ROOT / "runs" / "pkg-answers-embeddings" / "embeddings-shipped.npz"
TENANT = "acme"


class PkgAnswerAdapter:
    """The real model behind the product's retrieval, one answer per case."""

    name = "pkg-live"

    def __init__(self, client=None, corpus=None, vectors=None) -> None:
        self._client = client
        self._corpus = corpus if corpus is not None else load_corpus(CORPUS)
        self._vectors = vectors if vectors is not None else load_vectors(VECTORS)
        self._shared: AsyncQdrantClient | None = None
        self._lock = asyncio.Lock()

    def _model(self):
        if self._client is not None:
            return self._client
        from reliable_agents_labs.models import build_model_client

        return build_model_client("answer_model")

    async def _index(self, edits: dict[str, str]) -> AsyncQdrantClient:
        rows = [
            dict(r, summary=edits.get(r["name"], r["summary"])) for r in self._corpus
        ]
        qdrant = AsyncQdrantClient(":memory:")
        await ingest(qdrant, TENANT, rows, RecordedEmbedder(self._vectors, "doc"))
        return qdrant

    async def _index_for(self, edits: dict[str, str]) -> AsyncQdrantClient:
        if edits:
            return await self._index(edits)
        async with self._lock:
            if self._shared is None:
                self._shared = await self._index({})
            return self._shared

    async def run(self, case: EvalCase, trial: int) -> Trace:
        started = time.perf_counter()
        captured: list[dict[str, Any]] = []
        answer, cited, error = "", [], None
        try:
            qdrant = await self._index_for(case.input.get("summary_edits", {}))
            result = await ask_rag_agent_for_tenant(
                TENANT,
                case.input["query"],
                qdrant=qdrant,
                embedder=RecordedEmbedder(self._vectors, "query"),
                model_client=self._model(),
                limit=3,
                on_retrieval=captured.extend,
            )
            answer, cited = result.answer, list(result.cited_packages)
        except Exception as exc:  # noqa: BLE001 - a failed run is a result
            error = f"{type(exc).__name__}: {exc}"
        retrieved = [
            {
                "rank": rank,
                "name": item["name"],
                "summary": item["summary"],
                "score": round(float(item["score"]), 4),
            }
            for rank, item in enumerate(captured, 1)
        ]
        return Trace(
            case_id=case.case_id,
            trial=trial,
            adapter=self.name,
            answer=answer,
            error=error,
            latency_s=round(time.perf_counter() - started, 3),
            retrieved=retrieved,
            cited=cited,
        )


# ------------------------------------------------------------ scripted models

_CONTEXT = re.compile(r"Package information:\n(.*)\n\nQuestion:", re.DOTALL)


def _retrieved(user: str) -> list[tuple[str, str]]:
    block = _CONTEXT.search(user)
    rows = (block.group(1) if block else "").splitlines()
    return [tuple(r[2:].split(": ", 1)) for r in rows if r.startswith("- ")]


class ScriptedRagClient:
    """A model that answers from the retrieved text, and can be told to misbehave in
    exactly one way. `mode` is one of `faithful`, `cites_unretrieved`,
    `names_outside`, `states_version`, `cites_nothing`, `ignores_context` or
    `invalid_json`. `names` lists every package in the index, so the script can pick
    one that was not retrieved."""

    def __init__(self, mode: str = "faithful", names: list[str] | None = None) -> None:
        self.mode, self.names = mode, names or []

    async def generate(self, *, system, user, tools=None, history=None):
        found = _retrieved(user)
        text = "According to the package information: " + " ".join(
            f"{n}: {s}" for n, s in found
        )
        cited = [n for n, _ in found]
        seen = {n for n, _ in found}
        outside = next(
            (n for n in self.names if n not in seen and n.lower() not in user.lower()),
            "numpy",
        )
        if self.mode == "cites_unretrieved":
            cited = [outside]
        elif self.mode == "names_outside":
            text += f" You may also like {outside}."
        elif self.mode == "states_version":
            text += " The latest version is 9.9.9."
        elif self.mode == "cites_nothing":
            cited = []
        elif self.mode == "ignores_context":
            text = "From my own knowledge, this is a widely used package."
        elif self.mode == "invalid_json":
            return _result("Sorry, I cannot answer that.")
        payload = {"answer": text, "cited_packages": cited}
        return _result(json.dumps(payload))


def _result(text: str) -> ModelResult:
    return ModelResult(
        text=text,
        input_tokens=0,
        output_tokens=0,
        model_id="scripted",
        provider="scripted",
        tool_calls=[],
    )
