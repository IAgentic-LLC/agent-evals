"""A leaderboard that refuses (chapter 26).

A leaderboard is an evaluation of evaluations, so it needs its own rules. This one is a
static report built from run folders. It admits a run only if its manifest says where it
came from and the run is comparable with the others, it shows an interval beside every
number, it gives each entry a range of ranks and not a rank, and an entry that broke an
invariant is shown as blocked and is never ranked. It never adds the columns into one
score.
"""

import html
import itertools
import json
import random
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from pydantic import BaseModel

from agent_evals import compare, cost, reliability
from agent_evals.answer_graders import acted_on_the_right_customer
from agent_evals.graders import forbidden_actions_taken
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase
from agent_evals.stats import percentile

REQUIRED_FIELDS = (
    "adapter",
    "model.model_id",
    "dataset.sha256",
    "dataset.cases",
    "trials",
    "harness.commit",
    "harness.dirty",
)
DRAWS = 2000


class Candidate(BaseModel):
    run: str
    label: str


class OtherSet(BaseModel):
    dataset: str
    runs: dict[str, str]


class Board(BaseModel):
    name: str
    dataset: str
    min_trials: int = 3
    prices: str
    entries: list[Candidate]
    first_look: list[str] = []
    other_sets: list[OtherSet] = []


def load_board(path: str | Path) -> Board:
    return Board.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf8")))


def _dig(data: dict, dotted: str):
    for part in dotted.split("."):
        if not isinstance(data, dict) or part not in data:
            return None
        data = data[part]
    return data


def dataset_hash(path: Path) -> str:
    import hashlib

    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------- admission


def why_refused(
    run: Path, cases: dict[str, EvalCase], want_hash: str, min_trials: int
) -> list[str]:
    """Every reason the run cannot go on the board. An empty list means it can."""
    manifest_path = run / "manifest.json"
    if not manifest_path.exists():
        return ["no manifest"]
    manifest = json.loads(manifest_path.read_text(encoding="utf8"))
    reasons = []
    missing = [f for f in REQUIRED_FIELDS if _dig(manifest, f) is None]
    if missing:
        reasons.append("manifest lacks " + ", ".join(missing))
    if _dig(manifest, "harness.dirty"):
        reasons.append("harness had uncommitted changes")
    trials = manifest.get("trials") or 0
    if trials < min_trials:
        reasons.append(
            f"{trials} trial{'s' * (trials != 1)}, the board needs {min_trials}"
        )
    got = _dig(manifest, "dataset.sha256")
    if got is not None and got != want_hash:
        reasons.append(f"another dataset ({got[:6]}, not {want_hash[:6]})")
    traces_path = run / "traces.jsonl"
    if not traces_path.exists():
        return [*reasons, "no traces"]
    traces = read_traces(traces_path)
    if any(t.ledger_at_start > 0 for t in traces):
        left = sum(t.ledger_at_start > 0 for t in traces)
        reasons.append(f"{left} runs began with leftover state")
    seen = {t.case_id for t in traces}
    if got == want_hash and seen != set(cases):
        reasons.append(f"{len(set(cases) - seen)} tickets have no runs")
    return reasons


# ------------------------------------------------------------------------ rows


@dataclass
class Row:
    label: str
    run: str
    model: str
    trials: int
    per_ticket: dict[str, float]
    met: float
    interval: tuple[float, float]
    reliable: float
    customer: int
    forbidden: int
    errors: int
    runs: int
    cost_per_success: float | None
    p50: float | None
    p95: float | None
    rank_range: tuple[int, int] = (0, 0)
    first: float = 0.0
    passes: list[float] = field(default_factory=list)

    @property
    def blocked(self) -> bool:
        return self.customer > 0 or self.forbidden > 0


def _row(
    entry: Candidate,
    run: Path,
    cases: dict[str, EvalCase],
    prices: dict[str, cost.Price],
) -> Row:
    manifest = json.loads((run / "manifest.json").read_text(encoding="utf8"))
    traces = read_traces(run / "traces.jsonl")
    model = _dig(manifest, "model.model_id")
    by_ticket: dict[str, list[bool]] = {}
    for t in traces:
        by_ticket.setdefault(t.case_id, []).append(
            reliability.success(cases[t.case_id], t)
        )
    shares = {k: sum(v) / len(v) for k, v in by_ticket.items()}
    values = [shares[k] for k in sorted(shares)]
    met_runs = sum(sum(v) for v in by_ticket.values())
    metered = all("thinking_tokens" in t.usage for t in traces)
    price = prices.get(model)
    per_success = None
    if metered and price is not None and met_runs:
        per_success = sum(cost.run_cost(t, price) for t in traces) / met_runs
    lat = [t.latency_s for t in traces if t.latency_s is not None]
    trials = manifest["trials"]
    passes = []
    for k in range(1, trials + 1):
        one = [t for t in traces if t.trial == k]
        passes.append(
            sum(reliability.success(cases[t.case_id], t) for t in one) / len(one)
        )
    return Row(
        label=entry.label,
        run=entry.run,
        model=model,
        trials=trials,
        per_ticket=shares,
        met=sum(values) / len(values),
        interval=compare.cluster_interval(
            compare.mean, values, resamples=DRAWS, seed=0
        ),
        reliable=sum(all(v) for v in by_ticket.values()) / len(by_ticket),
        customer=sum(
            not acted_on_the_right_customer(cases[t.case_id], t) for t in traces
        ),
        forbidden=sum(
            bool(forbidden_actions_taken(cases[t.case_id], t)) for t in traces
        ),
        errors=sum(t.error is not None for t in traces),
        runs=len(traces),
        cost_per_success=per_success,
        p50=percentile(lat, 0.5) if lat else None,
        p95=percentile(lat, 0.95) if lat else None,
        passes=passes,
    )


def build(root: Path, board: Board):
    """Sort the board's runs into admitted rows and refused runs with their reasons."""
    dataset = root / board.dataset
    cases = {c.case_id: c for c in load_cases(dataset)}
    want = dataset_hash(dataset)
    prices = cost.load_prices(root / board.prices)
    rows: list[Row] = []
    refused: list[tuple[Candidate, list[str]]] = []
    for entry in board.entries:
        run = root / "runs" / entry.run
        reasons = why_refused(run, cases, want, board.min_trials)
        if reasons:
            refused.append((entry, reasons))
        else:
            rows.append(_row(entry, run, cases, prices))
    return rows, refused, cases


# ----------------------------------------------------------------------- ranks


def _rank(values: list[float]) -> list[int]:
    return [1 + sum(v > x for v in values) for x in values]


def add_rank_ranges(rows: list[Row], draws: int = DRAWS, seed: int = 0) -> None:
    """For each row, the 2.5th and 97.5th percentile of its rank over redraws of the
    tickets, and the share of redraws in which it ranks first (ties count for all)."""
    if not rows:
        return
    tickets = sorted(rows[0].per_ticket)
    rng = random.Random(seed)
    ranks: list[list[int]] = [[] for _ in rows]
    for _ in range(draws):
        pick = [tickets[rng.randrange(len(tickets))] for _ in tickets]
        means = [sum(r.per_ticket[k] for k in pick) / len(pick) for r in rows]
        for i, rank in enumerate(_rank(means)):
            ranks[i].append(rank)
    for row, seen in zip(rows, ranks, strict=True):
        seen.sort()
        row.rank_range = (seen[int(0.025 * draws)], seen[int(0.975 * draws) - 1])
        row.first = sum(r == 1 for r in seen) / draws


def rank_order(rows: list[Row]) -> list[Row]:
    return sorted(rows, key=lambda r: (-r.met, r.label))


# --------------------------------------------------------------------- tables


def _pct(x: float) -> str:
    return f"{100 * x:.0f}"


def _dollars(x: float | None) -> str:
    return "n/a" if x is None else f"${x:.4f}"


def _secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.1f}"


def render_admit(refused: list[tuple[Candidate, list[str]]], rows: list[Row]) -> str:
    lines = [f"{'run':<30}{'verdict':<10}why"]
    items = [(r.run, "admitted", []) for r in rows] + [
        (e.run, "refused", why) for e, why in refused
    ]
    for run, verdict, why in sorted(items):
        first = why[0] if why else ""
        lines.append(f"{run:<30}{verdict:<10}{first}".rstrip())
        for more in why[1:]:
            lines.append(f"{'':<40}{more}")
    lines.append(f"{len(rows)} admitted, {len(refused)} refused")
    return "\n".join(lines) + "\n"


def render_board(
    rows: list[Row],
    min_met: float | None = None,
    sort: str = "met",
) -> str:
    ranked = [r for r in rows if not r.blocked]
    blocked = [r for r in rows if r.blocked]
    add_rank_ranges(ranked)
    shown = [r for r in ranked if min_met is None or 100 * r.met >= min_met]
    order = {
        "met": lambda r: (-r.met, r.label),
        "cost": lambda r: (
            r.cost_per_success is None,
            r.cost_per_success or 0,
            r.label,
        ),
        "p95": lambda r: (r.p95 is None, r.p95 or 0, r.label),
    }[sort]
    shown.sort(key=order)
    head = (
        f"{'rank':<6}{'entry':<24}{'met%':>5}{'interval':>10}{'steady':>7}"
        f"{'$/success':>11}{'p95 s':>7}"
    )
    lines = [head]
    for r in shown:
        lo, hi = r.rank_range
        rank = str(lo) if lo == hi else f"{lo}-{hi}"
        span = f"{100 * r.interval[0]:.0f} to {100 * r.interval[1]:.0f}"
        lines.append(
            f"{rank:<6}{r.label:<24}{_pct(r.met):>5}{span:>10}"
            f"{_pct(r.reliable):>7}{_dollars(r.cost_per_success):>11}{_secs(r.p95):>7}"
        )
    if blocked:
        lines.append("")
        lines.append("BLOCKED, not ranked: an invariant was broken")
        lines.append(f"{'entry':<30}{'customer':>9}{'forbidden':>10}  runs")
        for r in sorted(blocked, key=lambda r: r.label):
            lines.append(f"{r.label:<30}{r.customer:>9}{r.forbidden:>10}  {r.runs}")
    lines.append("")
    lines += textwrap.wrap(
        "rank is the 95% range over 2000 redraws of the tickets. steady is the "
        "share of tickets met in every trial. interval is a 95% interval from "
        "redrawing tickets.",
        width=78,
    )
    return "\n".join(lines) + "\n"


def render_ranks(rows: list[Row]) -> str:
    ranked = rank_order([r for r in rows if not r.blocked])
    add_rank_ranges(ranked)
    lines = [f"{'entry':<24}{'met%':>5}{'best':>6}{'worst':>7}{'first':>8}"]
    for r in ranked:
        lines.append(
            f"{r.label:<24}{_pct(r.met):>5}{r.rank_range[0]:>6}{r.rank_range[1]:>7}"
            f"{_pct(r.first) + '%':>8}"
        )
    lines.append("first is the share of redraws in which the entry ranks first")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------- one run per agent


def render_first_look(rows: list[Row], labels: list[str]) -> str:
    """The leaderboard most people build: one pass per entry, ranked by the met rate.
    Then the same board from each of the other passes, and every one-pass board."""
    picked = [r for r in rows if r.label in labels]
    picked.sort(key=lambda r: labels.index(r.label))
    n_pass = min(r.trials for r in picked)
    lines = [
        f"{'entry':<24}" + "".join(f"{f'pass {k + 1}':>13}" for k in range(n_pass))
    ]
    ranks = [_rank([r.passes[k] for r in picked]) for k in range(n_pass)]
    order = sorted(range(len(picked)), key=lambda i: (ranks[0][i], picked[i].label))
    for i in order:
        cells = "".join(
            f"{f'{100 * picked[i].passes[k]:.1f} ({ranks[k][i]})':>13}"
            for k in range(n_pass)
        )
        lines.append(f"{picked[i].label:<24}{cells}")
    lines.append("each cell is the met% of one pass over the tickets, and its rank")
    orders: dict[tuple[int, ...], int] = {}
    first = [0] * len(picked)
    boards = list(itertools.product(range(n_pass), repeat=len(picked)))
    for combo in boards:
        vals = [picked[i].passes[combo[i]] for i in range(len(picked))]
        rk = _rank(vals)
        orders[tuple(rk)] = orders.get(tuple(rk), 0) + 1
        for i, x in enumerate(rk):
            first[i] += x == 1
    lines.append("")
    lines.append(
        f"{len(boards)} boards are possible: each entry run once, with any of "
        f"its {n_pass} passes"
    )
    lines.append(f"{len(orders)} different orders appear among them")
    lines.append(f"{'entry':<24}{'ranked first':>13}{'best':>6}{'worst':>7}")
    for i in order:
        seen = [
            _rank([picked[j].passes[c[j]] for j in range(len(picked))])[i]
            for c in boards
        ]
        lines.append(
            f"{picked[i].label:<24}{f'{100 * first[i] / len(boards):.0f}%':>13}"
            f"{min(seen):>6}{max(seen):>7}"
        )
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------- dataset versions


def render_versions(rows: list[Row], root: Path, first: str, second: str) -> str:
    """The same runs scored against two versions of the dataset. A board that mixed
    the two would rank on a change in the labels and not on a change in the agents."""
    lines = [f"{'entry':<24}{'v1 met%':>9}{'v2 met%':>9}{'change':>8}"]
    sets = [
        {c.case_id: c for c in load_cases(root / first)},
        {c.case_id: c for c in load_cases(root / second)},
    ]
    scored = []
    for r in rows:
        traces = read_traces(root / "runs" / r.run / "traces.jsonl")
        met = [
            sum(reliability.success(cases[t.case_id], t) for t in traces) / len(traces)
            for cases in sets
        ]
        scored.append((r.label, met[0], met[1]))
    for label, a, b in sorted(scored, key=lambda x: -x[1]):
        lines.append(f"{label:<24}{_pct(a):>9}{_pct(b):>9}{100 * (b - a):>+8.1f}")
    changed = sum(
        sets[0][k].expected != sets[1][k].expected for k in sets[0] if k in sets[1]
    )
    lines.append(
        f"{changed} of {len(sets[0])} tickets have a different expected result"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- report


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


PAGE_CSS = """
:root{--ink:#1F2937;--muted:#6B7280;--line:#D1D5DB;--bg:#FFFFFF;--blue:#2563EB;
--red:#DC2626;--grey:#9CA3AF}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--ink:#E5E7EB;
--muted:#9CA3AF;--line:#374151;--bg:#111827;--blue:#60A5FA;--red:#F87171;--grey:#6B7280}}
body{font:15px/1.5 system-ui,sans-serif;max-width:52rem;margin:2rem auto;padding:0 1rem;
color:var(--ink);background:var(--bg)}
table{border-collapse:collapse;width:100%}
th,td{padding:.35rem .6rem;border-bottom:1px solid var(--line);text-align:left}
th{font-size:.8rem;text-transform:uppercase;letter-spacing:.04em;color:var(--muted)}
td:nth-child(n+3),th:nth-child(n+3){font-variant-numeric:tabular-nums}
svg{max-width:100%;height:auto}svg text{fill:var(--ink);font-size:11px}
.note{color:var(--muted);font-size:.9rem}
"""


def _chart(ranked: list[Row], blocked: list[Row]) -> str:
    """Cost of a success against the met rate, one interval per entry, blocked entries as
    crosses. Both axes are labeled and every mark sits inside the drawing."""
    pts = [r for r in [*ranked, *blocked] if r.cost_per_success is not None]
    if not pts:
        return ""
    left, right, top, bottom = 56, 600, 16, 214
    x_max = 1000 * max(r.cost_per_success for r in pts) * 1.15
    x_max = max(1.0, float(int(x_max) + 1))

    def x(dollars: float) -> float:
        return left + (right - left) * 1000 * dollars / x_max

    def y(share: float) -> float:
        return bottom - (bottom - top) * (share - 0.4) / 0.6

    parts = [
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{bottom}" stroke="var(--ink)"/>',
        f'<line x1="{left}" y1="{bottom}" x2="{right}" y2="{bottom}" stroke="var(--ink)"/>',
    ]
    for tick in range(40, 101, 10):
        ty = y(tick / 100)
        parts.append(
            f'<line x1="{left - 4}" y1="{ty:.1f}" x2="{left}" y2="{ty:.1f}" '
            f'stroke="var(--ink)"/><text x="{left - 8}" y="{ty + 4:.1f}" '
            f'text-anchor="end">{tick}</text>'
        )
    for tick in range(int(x_max) + 1):
        tx = left + (right - left) * tick / x_max
        parts.append(
            f'<line x1="{tx:.1f}" y1="{bottom}" x2="{tx:.1f}" y2="{bottom + 4}" '
            f'stroke="var(--ink)"/><text x="{tx:.1f}" y="{bottom + 16}" '
            f'text-anchor="middle">{tick}</text>'
        )
    parts.append(
        f'<text x="{(left + right) / 2}" y="{bottom + 32}" text-anchor="middle">'
        "cost of a success, in thousandths of a dollar</text>"
        f'<text transform="translate(12 {(top + bottom) / 2}) rotate(-90)" '
        'text-anchor="middle">met rate (%)</text>'
    )
    placed: list[tuple[float, float]] = []

    def label(px: float, py: float, text: str, color: str) -> None:
        ly = py - 9
        for ox, oy in placed:
            if abs(ox - px) < 90 and abs(oy - ly) < 13:
                ly = oy + 13
        placed.append((px, ly))
        parts.append(
            f'<text x="{px + 8:.1f}" y="{ly:.1f}" style="fill:{color}">'
            f"{html.escape(text)}</text>"
        )

    for r in sorted(ranked, key=lambda r: (r.cost_per_success, -r.met)):
        px = x(r.cost_per_success)
        parts.append(
            f'<line x1="{px:.1f}" y1="{y(r.interval[0]):.1f}" x2="{px:.1f}" '
            f'y2="{y(r.interval[1]):.1f}" stroke="var(--grey)" stroke-width="2"/>'
            f'<circle cx="{px:.1f}" cy="{y(r.met):.1f}" r="4.5" fill="var(--blue)"/>'
        )
        label(px, y(r.met), r.label, "var(--ink)")
    for r in sorted(blocked, key=lambda r: (r.cost_per_success, -r.met)):
        px, py = x(r.cost_per_success), y(r.met)
        parts.append(
            f'<path d="M{px - 4:.1f} {py - 4:.1f}l8 8m0 -8l-8 8" stroke="var(--red)" '
            'stroke-width="2"/>'
        )
        label(px, py, f"{r.label} (blocked)", "var(--red)")
    return (
        '<svg viewBox="0 0 640 250" role="img" aria-label="Met rate against the cost of a '
        'success, with an interval on each ranked entry and a cross on each blocked one">'
        + "".join(parts)
        + "</svg>"
    )


def render_html(board: Board, rows: list[Row], refused, title: str) -> str:
    """A single static page: the table, the chart of cost against the met rate, the
    blocked entries, and the refused runs with their reasons."""
    ranked = [r for r in rows if not r.blocked]
    blocked = [r for r in rows if r.blocked]
    add_rank_ranges(ranked)
    ranked.sort(key=lambda r: (-r.met, r.label))
    body = []
    for r in ranked:
        lo, hi = r.rank_range
        rank = str(lo) if lo == hi else f"{lo}-{hi}"
        body.append(
            f"<tr><td>{rank}</td><td>{html.escape(r.label)}</td>"
            f"<td>{100 * r.met:.0f}</td>"
            f"<td>{100 * r.interval[0]:.0f} to {100 * r.interval[1]:.0f}</td>"
            f"<td>{100 * r.reliable:.0f}</td><td>{_dollars(r.cost_per_success)}</td>"
            f"<td>{_secs(r.p95)}</td></tr>"
        )
    blocked_items = "".join(
        f"<li>{html.escape(r.label)}: {_plural(r.customer, 'run')} acted on another "
        f"customer, {_plural(r.forbidden, 'run')} took a forbidden action</li>"
        for r in blocked
    )
    refused_items = "".join(
        f"<li><code>{html.escape(e.run)}</code>: {html.escape('; '.join(why))}</li>"
        for e, why in refused
    )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f"<title>{html.escape(title)}</title><style>{PAGE_CSS}</style></head><body>"
        f"<h1>{html.escape(title)}</h1>"
        f'<p class="note">Dataset <code>{html.escape(board.dataset)}</code>, at least '
        f"{board.min_trials} trials of every ticket. A rank is the 95% range over "
        "redraws of the tickets. The columns are never added into one score.</p>"
        "<table><tr><th>rank</th><th>entry</th><th>met %</th><th>interval</th>"
        "<th>steady %</th><th>$ per success</th><th>p95 s</th></tr>"
        + "".join(body)
        + "</table>"
        '<p class="note">Steady is the share of tickets met in every trial. The '
        "interval redraws whole tickets.</p>"
        "<h2>Cost against the met rate</h2>"
        + _chart(ranked, blocked)
        + (f"<h2>Blocked, not ranked</h2><ul>{blocked_items}</ul>" if blocked else "")
        + (f"<h2>Refused</h2><ul>{refused_items}</ul>" if refused else "")
        + "</body></html>"
    )


# ------------------------------------------------------------- another dataset


def render_sets(rows: list[Row], root: Path, other: OtherSet, board: Board) -> str:
    """The same configurations on a second set of tickets. A score belongs to a dataset,
    so the gap between two entries on one set says little about the gap on another."""
    held = {c.case_id: c for c in load_cases(root / board.dataset)}
    fresh = {c.case_id: c for c in load_cases(root / other.dataset)}
    by = {r.label: r for r in rows}
    lines = [f"{'entry':<24}{'held-out':>10}{'this set':>10}"]
    got = []
    for label, run in other.runs.items():
        traces = read_traces(root / "runs" / run / "traces.jsonl")
        met = sum(reliability.success(fresh[t.case_id], t) for t in traces) / len(
            traces
        )
        got.append((label, by[label].met, met))
    for label, a, b in got:
        lines.append(f"{label:<24}{_pct(a):>10}{_pct(b):>10}")
    if len(got) == 2:
        lines.append(
            f"{'gap between them':<24}{100 * (got[1][1] - got[0][1]):>+10.0f}"
            f"{100 * (got[1][2] - got[0][2]):>+10.0f}"
        )
    lines.append(f"held-out is {len(held)} tickets, this set is {len(fresh)}")
    return chr(10).join(lines) + chr(10)
