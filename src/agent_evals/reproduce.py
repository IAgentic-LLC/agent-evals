"""Rerun a recorded run yourself, with your own key (appendix A).

Every run folder has a manifest that says what produced it: the adapter, the dataset, the
number of trials, how many runs were in flight and which model. This module turns a
manifest back into the command that made the run, and says what the run cost me, so that
you can decide whether to spend the same on your own key. It never calls a model.
"""

import json
import textwrap
from pathlib import Path

from agent_evals import cost
from agent_evals.runner import read_traces

LIVE_PREFIXES = ("triage-live", "reorder-live", "pkg-live")


def manifest(root: Path, name: str) -> dict | None:
    path = root / "runs" / name / "manifest.json"
    return json.loads(path.read_text(encoding="utf8")) if path.exists() else None


def kind(data: dict | None) -> str:
    """What made the run: an adapter over a dataset, or one of the other commands."""
    if data is None:
        return "no manifest"
    if "adapter" in data and "dataset" in data and "trials" in data:
        return "adapter"
    if "judge_model" in data:
        return "judge"
    if "embedding_model" in data:
        return "retrieval"
    return "other"


def needs_key(data: dict) -> bool:
    return data["adapter"].startswith(LIVE_PREFIXES)


def estimate(
    root: Path, name: str, data: dict, prices: dict[str, cost.Price]
) -> tuple[float | None, int]:
    """Dollars for the recorded tokens at the prices in the file, and the run count.
    None when the run recorded no tokens or its model has no price."""
    path = root / "runs" / name / "traces.jsonl"
    if not path.exists():
        return None, 0
    traces = read_traces(path)
    price = prices.get((data.get("model") or {}).get("model_id", ""))
    if price is None or not all(t.usage for t in traces):
        return None, len(traces)
    return sum(cost.run_cost(t, price) for t in traces), len(traces)


def _dollars(x: float | None) -> str:
    if x is None:
        return "n/a"
    return f"${x:.2f}" if x >= 0.01 else f"${x:.4f}"


def render_list(root: Path, prices: dict[str, cost.Price], match: str = "") -> str:
    lines = [f"{'run':<30}{'model':<16}{'trials':>7}{'runs':>6}{'cost':>9}{'key':>6}"]
    total = 0.0
    shown = 0
    for folder in sorted((root / "runs").iterdir()):
        if match not in folder.name:
            continue
        data = manifest(root, folder.name)
        if kind(data) != "adapter":
            continue
        dollars, runs = estimate(root, folder.name, data, prices)
        total += dollars or 0.0
        shown += 1
        model = (data.get("model") or {}).get("model_id", "").removeprefix("gemini-")
        lines.append(
            f"{folder.name:<30}{model:<16}{data['trials']:>7}{runs:>6}"
            f"{_dollars(dollars):>9}{'yes' if needs_key(data) else 'no':>6}"
        )
    lines.append(
        f"{shown} runs, about {_dollars(total)} in all at the prices in the file"
    )
    return "\n".join(lines) + "\n"


def render_command(
    root: Path,
    name: str,
    prices: dict[str, cost.Price],
    answer_model_adapters: tuple[str, ...],
    out: str | None = None,
) -> str:
    data = manifest(root, name)
    what = kind(data)
    if what != "adapter":
        note = (
            f"{name} is not an adapter run ({what}). The chapter that made it "
            "shows its command."
        )
        return chr(10).join(textwrap.wrap(note, width=78)) + chr(10)
    out = out or f"runs/mine-{name.removeprefix('triage-')}"
    parts = [
        "uv run agent-evals run",
        f"--adapter {data['adapter']}",
        f"--dataset {data['dataset']['path']}",
        f"--trials {data['trials']}",
        f"--concurrency {data.get('concurrency', 1)}",
    ]
    model = (data.get("model") or {}).get("model_id")
    if model and data["adapter"] in answer_model_adapters:
        parts.append(f"--answer-model {model}")
    parts.append(f"--out {out}")
    if needs_key(data):
        parts.append("--env-file .env")
    lines = [f"{parts[0]} {parts[1]}"]
    for part in parts[2:]:
        lines[-1] += " " + chr(92)
        lines.append("  " + part)
    text = chr(10).join(lines) + chr(10)
    dollars, runs = estimate(root, name, data, prices)
    minutes = data.get("wall_seconds", 0) / 60
    flight = data.get("concurrency", 1)
    notes = [
        f"The recorded run made {runs} runs in {minutes:.1f} minutes, {flight} at a time."
    ]
    if dollars is not None:
        notes.append(
            f"Its tokens cost about {_dollars(dollars)} at the prices in "
            "config/prices.yaml. That is a floor, and prices change."
        )
    else:
        notes.append("It recorded no token counts, so I cannot estimate its cost.")
    if not needs_key(data):
        notes.append("This adapter needs no key.")
    elif model and data["adapter"] not in answer_model_adapters:
        notes.append(
            f"This adapter reads its model from config/models.yaml. The recorded run "
            f"used {model}, so check that file says the same."
        )
    for note in notes:
        text += chr(10).join(textwrap.wrap(note, width=78)) + chr(10)
    return text
