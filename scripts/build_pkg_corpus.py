"""Fetch the package corpus for the retrieval chapter (datasets/pkg_corpus_v1.jsonl), chapter 12.

Each row is a real PyPI package: its name, version and one-line summary, fetched through
the product's own `fetch_package_metadata` on the date in `fetched_at`. The summary is the
text the product embeds. The set is frozen once written: PyPI changes, and a later fetch
would be a different corpus, so it would be a new version.

The names are ones I chose, grouped by what they do, with several near neighbors in each
group on purpose (five HTTP clients, five plotting libraries), because a retrieval system
is hardest on packages that sound alike.

Usage: uv run python scripts/build_pkg_corpus.py
"""

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

from reliable_agents_labs.pypi import fetch_package_metadata

OUT = Path(__file__).resolve().parents[1] / "datasets" / "pkg_corpus_v1.jsonl"

GROUPS = {
    "http": [
        "requests",
        "httpx",
        "aiohttp",
        "urllib3",
        "httplib2",
        "websockets",
        "paramiko",
    ],
    "web": [
        "flask",
        "django",
        "fastapi",
        "starlette",
        "bottle",
        "tornado",
        "pyramid",
        "sanic",
    ],
    "data": ["numpy", "pandas", "polars", "scipy", "dask", "pyarrow", "xarray"],
    "ml": [
        "scikit-learn",
        "torch",
        "tensorflow",
        "xgboost",
        "lightgbm",
        "transformers",
        "keras",
    ],
    "plot": ["matplotlib", "seaborn", "plotly", "bokeh", "altair"],
    "parse": [
        "beautifulsoup4",
        "lxml",
        "html5lib",
        "pyyaml",
        "jsonschema",
        "orjson",
        "ujson",
        "xmltodict",
    ],
    "test": [
        "pytest",
        "hypothesis",
        "tox",
        "coverage",
        "faker",
        "responses",
        "pytest-cov",
    ],
    "model": ["pydantic", "attrs", "marshmallow", "cattrs", "dataclasses-json"],
    "db": ["sqlalchemy", "psycopg", "asyncpg", "peewee", "pymongo", "redis", "alembic"],
    "cli": ["click", "typer", "rich", "tqdm", "fire"],
    "async": ["celery", "rq", "dramatiq", "arq", "apscheduler", "trio", "anyio"],
    "files": [
        "pillow",
        "opencv-python",
        "imageio",
        "reportlab",
        "python-docx",
        "openpyxl",
        "pypdf",
    ],
    "dev": [
        "setuptools",
        "poetry",
        "pip",
        "virtualenv",
        "black",
        "ruff",
        "mypy",
        "flake8",
        "isort",
        "pylint",
    ],
    "security": ["cryptography", "pyjwt", "bcrypt", "passlib", "pynacl"],
    "cloud": [
        "boto3",
        "google-cloud-storage",
        "azure-storage-blob",
        "kubernetes",
        "docker",
    ],
    "misc": [
        "python-dateutil",
        "pytz",
        "arrow",
        "pendulum",
        "loguru",
        "structlog",
        "python-dotenv",
    ],
}


async def main() -> None:
    names = [(topic, name) for topic, group in GROUPS.items() for name in group]
    gate = asyncio.Semaphore(8)

    async def one(topic: str, name: str):
        async with gate:
            try:
                return topic, await fetch_package_metadata(name)
            except Exception as exc:  # noqa: BLE001 - report and skip
                print("skipped", name, type(exc).__name__)
                return topic, None

    results = await asyncio.gather(*(one(t, n) for t, n in names))
    fetched = datetime.now(UTC).date().isoformat()
    rows = [
        {
            "name": meta.name,
            "version": meta.version,
            "summary": meta.summary,
            "topic": topic,
            "fetched_at": fetched,
        }
        for topic, meta in results
        if meta is not None and meta.summary.strip()
    ]
    OUT.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
        encoding="utf8",
        newline="\n",
    )
    print(f"wrote {len(rows)} packages to {OUT}")


if __name__ == "__main__":
    asyncio.run(main())
