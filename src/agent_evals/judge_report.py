"""Read a judge's recorded verdicts against known truth and against my readings (chapter 15)."""

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from agent_evals.stats import wilson_interval


def load_rows(runs: list[str | Path]) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        text = (Path(run) / "judgments.jsonl").read_text(encoding="utf8")
        rows += [json.loads(x) for x in text.splitlines() if x.strip()]
    return rows


def _by_item(items):
    return {i["item_id"]: i for i in items}


def _keep(item, split):
    return split is None or item["split"] == split


def _cell(k: int, n: int) -> str:
    if n == 0:
        return "-"
    low, high = wilson_interval(k, n)
    return f"{k} of {n} ({100 * low:.0f}-{100 * high:.0f}%)"


def render_planted(items, rows, split: str | None = None) -> str:
    """How often the judge said `unsupported` on each kind of planted fault, and on the
    clean originals of those answers. Every verdict counts, both passes."""
    by_item = _by_item(items)
    faulty: dict[str, Counter] = defaultdict(Counter)
    clean = Counter()
    for r in rows:
        item = by_item[r["item_id"]]
        if not _keep(item, split) or r["verdict"] is None:
            continue
        said = r["verdict"] == "unsupported"
        if item["group"] == "planted":
            faulty[item["kind"]]["n"] += 1
            faulty[item["kind"]]["said"] += said
        elif item["clean_of_planted"]:
            clean["n"] += 1
            clean["said"] += said
    lines = [f"{'answers':<34}{'judge said unsupported':>34}"]
    for kind in ("praise", "fact", "capability"):
        c = faulty[kind]
        lines.append(f"{'with ' + kind + ' added':<34}{_cell(c['said'], c['n']):>34}")
    lines.append(
        f"{'the same answers, clean':<34}{_cell(clean['said'], clean['n']):>34}"
    )
    return "\n".join(lines) + "\n"


def render_real(items, rows, split: str | None = None) -> str:
    """The judge's verdicts on the real answers, by my reading of each."""
    by_item = _by_item(items)
    table: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        item = by_item[r["item_id"]]
        if item["group"] != "real" or not _keep(item, split) or r["verdict"] is None:
            continue
        table[item["reading"]]["n"] += 1
        table[item["reading"]]["said"] += r["verdict"] == "unsupported"
    lines = [f"{'my reading':<20}{'judge said unsupported':>40}"]
    for reading in ("supported", "borderline", "stretch"):
        c = table[reading]
        if c["n"]:
            lines.append(f"{reading:<20}{_cell(c['said'], c['n']):>40}")
    return "\n".join(lines) + "\n"


def render_retest(items, rows, split: str | None = None) -> str:
    """Do two passes over the same item give the same verdict?"""
    by_item = _by_item(items)
    verdicts: dict[str, dict[int, str | None]] = defaultdict(dict)
    for r in rows:
        if _keep(by_item[r["item_id"]], split):
            verdicts[r["item_id"]][r["pass"]] = r["verdict"]
    same = both = 0
    for v in verdicts.values():
        if len(v) == 2 and None not in v.values():
            both += 1
            same += len(set(v.values())) == 1
    return f"Same verdict in both passes: {_cell(same, both)}\n"


def render_cost(rows) -> str:
    calls = len(rows)
    errors = sum(1 for r in rows if r["verdict"] is None)
    tin = sum(r.get("input_tokens", 0) for r in rows)
    tout = sum(r.get("output_tokens", 0) for r in rows)
    secs = sum(r["seconds"] for r in rows)
    return (
        f"{'calls':<28}{calls:>10}\n"
        f"{'replies that were not JSON':<28}{errors:>10}\n"
        f"{'input tokens':<28}{tin:>10}\n"
        f"{'output tokens':<28}{tout:>10}\n"
        f"{'mean seconds per call':<28}{secs / calls:>10.1f}\n"
    )


def disagreements(items, rows, split: str | None = None) -> list[tuple[dict, str]]:
    """Real answers I read as supported that the judge called unsupported in every pass,
    with the first claim it marked unsupported."""
    by_item = _by_item(items)
    said: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        said[r["item_id"]].append(r)
    out = []
    for item_id, rs in said.items():
        item = by_item[item_id]
        if item["group"] != "real" or item["label"] != "supported":
            continue
        if not _keep(item, split) or not all(r["verdict"] == "unsupported" for r in rs):
            continue
        claims = [c["claim"] for c in rs[0]["claims"] if not c.get("supported", True)]
        out.append((item, claims[0] if claims else ""))
    return out
