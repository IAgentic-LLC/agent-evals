"""A run manifest: what produced a set of traces, so a number can be traced to its source."""

import hashlib
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml


def _source(package: str) -> dict[str, Any] | None:
    """Which code an installed package really is: its version and, for a package
    installed from git, the requested tag and the exact commit.
    """
    try:
        dist = metadata.distribution(package)
    except metadata.PackageNotFoundError:
        return None
    info: dict[str, Any] = {"version": dist.version}
    raw = dist.read_text("direct_url.json")
    vcs = json.loads(raw).get("vcs_info", {}) if raw else {}
    if vcs.get("requested_revision"):
        info["requested_revision"] = vcs["requested_revision"]
    if vcs.get("commit_id"):
        info["commit"] = vcs["commit_id"]
    return info


def _harness_state() -> dict[str, Any] | None:
    """The harness's own commit, and whether the tree had uncommitted changes."""

    def git(*args: str) -> str:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True
        ).stdout.strip()

    try:
        return {
            "commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain")),
        }
    except (OSError, subprocess.CalledProcessError):
        return None


def _model_config(path: str = "config/models.yaml") -> dict[str, Any] | None:
    """The model the live adapters will use, read the way the product reads it."""
    try:
        config = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    except OSError:
        return None
    role = config.get("answer_model", {})
    return {"provider": role.get("provider"), "model_id": role.get("model_id")}


def build_manifest(
    *,
    adapter: str,
    dataset: str | Path,
    cases: int,
    trials: int,
    concurrency: int,
    started_at: datetime,
    wall_seconds: float,
) -> dict[str, Any]:
    live = adapter.startswith("triage-live")
    return {
        "adapter": adapter,
        "model": _model_config() if live else None,
        "dataset": {
            "path": str(dataset),
            "sha256": hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
            "cases": cases,
        },
        "trials": trials,
        "concurrency": concurrency,
        "started_at": started_at.astimezone(UTC).isoformat(timespec="seconds"),
        "wall_seconds": round(wall_seconds, 2),
        "harness": _harness_state(),
        "packages": {
            "triage-app": _source("triage-app"),
            "reliable-agents-labs": _source("reliable-agents-labs"),
        },
    }


def now() -> datetime:
    return datetime.now(UTC)
