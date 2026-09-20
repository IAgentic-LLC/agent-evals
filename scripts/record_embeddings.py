"""Record the embeddings for the retrieval chapter (chapter 12). This is the only live step.

It embeds every corpus summary as a document and every query as a query, two ways:
  shipped   the product's own embedding client, no task type
  typed     Gemini's native endpoint, RETRIEVAL_DOCUMENT for summaries and
            RETRIEVAL_QUERY for questions, as Google's documentation recommends

Everything after this step reads the recording, so the scores can be reproduced with no
key and no network.

Usage:
    uv run python scripts/record_embeddings.py OUT_DIR --queries FILE --env-file ENV
"""

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from reliable_agents_labs.models import build_embedding_client

from agent_evals.manifest import harness_state, now
from agent_evals.retrieval import (
    NativeEmbedder,
    load_corpus,
    record_vectors,
    save_vectors,
)
from agent_evals.runner import load_cases

ROOT = Path(__file__).resolve().parents[1]
CORPUS = ROOT / "datasets" / "pkg_corpus_v1.jsonl"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


async def main(out: Path, queries_path: Path) -> None:
    QUERIES = queries_path
    corpus = load_corpus(CORPUS)
    cases = load_cases(QUERIES)
    queries = [c.input["query"] for c in cases]
    # A case may ask against an index in which a summary was edited (chapter 13).
    # The edited texts are embedded as documents too.
    edits = sorted(
        {t for c in cases for t in c.input.get("summary_edits", {}).values()}
    )
    harness, started = harness_state(), now()
    clock = time.perf_counter()
    shipped = build_embedding_client().embed
    kinds = {
        "shipped": (shipped, shipped),
        "typed": (
            NativeEmbedder("RETRIEVAL_DOCUMENT").embed,
            NativeEmbedder("RETRIEVAL_QUERY").embed,
        ),
    }
    out.mkdir(parents=True, exist_ok=True)
    for kind, (doc_embed, query_embed) in kinds.items():
        keys, vectors = await record_vectors(
            corpus + [{"summary": t} for t in edits], queries, doc_embed, query_embed
        )
        save_vectors(out / f"embeddings-{kind}.npz", keys, vectors)
        print(kind, len(keys), "vectors of", len(vectors[0]))
    manifest = {
        "embedding_model": "gemini-embedding-001",
        "kinds": {
            "shipped": "product client, OpenAI-compatible endpoint, no task type",
            "typed": "native embedContent, RETRIEVAL_DOCUMENT and RETRIEVAL_QUERY",
        },
        "corpus": {
            "path": str(CORPUS.name),
            "sha256": sha(CORPUS),
            "packages": len(corpus),
        },
        "queries": {
            "path": str(QUERIES.name),
            "sha256": sha(QUERIES),
            "count": len(queries),
            "summary_edits": len(edits),
        },
        "started_at": started.isoformat(timespec="seconds"),
        "wall_seconds": round(time.perf_counter() - clock, 2),
        "harness": harness,
    }
    (out / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf8"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--env-file")
    args = parser.parse_args()
    if args.env_file:
        load_dotenv(args.env_file)
    asyncio.run(main(args.out, args.queries))
