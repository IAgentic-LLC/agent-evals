"""What a run costs, how long it takes, and what an evaluation costs (chapter 22).

Prices are data, not code: they live in a file that names its source and the date it was
checked, and the commands take that file. Tokens come from the provider's own count for
each call, summed over a run. Output tokens may leave out thinking tokens, so every
dollar figure here is a floor.
"""

import random
from pathlib import Path
from typing import Any

import yaml

from agent_evals import compare, reliability
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import percentile, wilson_interval

Price = tuple[float, float]


def load_prices(path: str | Path) -> dict[str, Price]:
    """{model id: (dollars per million input tokens, per million output tokens)}."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf8"))
    return {
        model: (float(p["input"]), float(p["output"]))
        for model, p in data["models"].items()
    }


def run_cost(trace: Trace, price: Price) -> float:
    """Dollars for one run, from its recorded tokens. A run recorded before tokens were
    recorded has no cost, and asking for it is an error."""
    if not trace.usage:
        raise ValueError(f"{trace.case_id} trial {trace.trial} has no recorded tokens")
    return (
        trace.usage["input_tokens"] * price[0] + trace.usage["output_tokens"] * price[1]
    ) / 1_000_000


def outcome(case: EvalCase, trace: Trace) -> str:
    """How a run ended: it met the required actions, it hit a tool loop, a handoff loop,
    or it finished without taking a required action."""
    if trace.error is None:
        return "met" if reliability.success(case, trace) else "missed"
    return "handoff loop" if trace.error.startswith("HandoffLoop") else "tool loop"


OUTCOMES = ("met", "missed", "tool loop", "handoff loop")


def _latencies(traces: list[Trace]) -> list[float]:
    return [t.latency_s for t in traces if t.latency_s is not None]


def latency_interval(
    traces: list[Trace], q: float, resamples: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% interval for a latency percentile, redrawing whole tickets."""
    by_ticket: dict[str, list[float]] = {}
    for t in traces:
        if t.latency_s is not None:
            by_ticket.setdefault(t.case_id, []).append(t.latency_s)
    tickets = list(by_ticket.values())
    rng = random.Random(seed)
    draws = []
    for _ in range(resamples):
        picked = [x for v in rng.choices(tickets, k=len(tickets)) for x in v]
        draws.append(percentile(picked, q))
    draws.sort()
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def _k(x: float) -> str:
    return f"{x / 1000:.1f}k"


def render_models(
    cases: dict[str, EvalCase],
    conditions: list[tuple[str, str, list[Trace]]],
    prices: dict[str, Price],
) -> str:
    """One row per condition: (label, model id, traces)."""
    lines = [
        f"{'condition':<19}{'runs':>5}{'met':>5}{'tokens in/out':>15}"
        + f"{'$/run':>8}{'$/met':>8}{'p50 s':>7}{'p95 s':>7}"
    ]
    for label, model, traces in conditions:
        met = sum(outcome(cases[t.case_id], t) == "met" for t in traces)
        n = len(traces)
        t_in = sum(t.usage["input_tokens"] for t in traces) / n
        t_out = sum(t.usage["output_tokens"] for t in traces) / n
        total = sum(run_cost(t, prices[model]) for t in traces)
        lat = _latencies(traces)
        lines.append(
            f"{label:<19}{n:>5}{met:>5}{f'{_k(t_in)}/{_k(t_out)}':>15}"
            f"{total / n:>8.4f}{total / met:>8.4f}"
            f"{percentile(lat, 0.5):>7.1f}{percentile(lat, 0.95):>7.1f}"
        )
    return "\n".join(lines) + "\n"


def render_spend(cases: dict[str, EvalCase], traces: list[Trace], price: Price) -> str:
    """Where the money goes: for each way a run can end, the runs, the average cost of one
    and its share of the total."""
    rows: dict[str, list[float]] = {o: [] for o in OUTCOMES}
    for t in traces:
        rows[outcome(cases[t.case_id], t)].append(run_cost(t, price))
    total = sum(sum(v) for v in rows.values())
    lines = [f"{'run ended':<14}{'runs':>6}{'$ per run':>11}{'share of spend':>16}"]
    for name in OUTCOMES:
        v = rows[name]
        if not v:
            lines.append(f"{name:<14}{0:>6}{'-':>11}{'-':>16}")
            continue
        lines.append(
            f"{name:<14}{len(v):>6}{sum(v) / len(v):>11.4f}"
            f"{f'{100 * sum(v) / total:.0f}%':>16}"
        )
    return "\n".join(lines) + "\n"


def render_latency(cases: dict[str, EvalCase], traces: list[Trace]) -> str:
    """Latency of one condition, overall and by how the run ended, with intervals from
    redrawing tickets for the percentiles."""
    lines = [
        f"{'run ended':<14}{'runs':>6}{'p50 s':>8}{'p95 s':>8}{'p95 interval':>16}"
    ]
    groups = [("all", traces)] + [
        (o, [t for t in traces if outcome(cases[t.case_id], t) == o]) for o in OUTCOMES
    ]
    for name, group in groups:
        lat = _latencies(group)
        if not lat:
            continue
        low, high = latency_interval(group, 0.95)
        lines.append(
            f"{name:<14}{len(group):>6}{percentile(lat, 0.5):>8.1f}"
            f"{percentile(lat, 0.95):>8.1f}{f'{low:.1f} to {high:.1f}':>16}"
        )
    return "\n".join(lines) + "\n"


def pareto(points: list[tuple[str, float, float]]) -> list[str]:
    """The labels not beaten by another point that is at least as cheap and at least as
    good, and better in one. Each point is (label, cost, quality)."""
    best = []
    for label, cost, quality in points:
        beaten = any(
            c <= cost and q >= quality and (c < cost or q > quality)
            for other, c, q in points
            if other != label
        )
        if not beaten:
            best.append(label)
    return best


def render_frontier(
    cases: dict[str, EvalCase],
    conditions: list[tuple[str, str, list[Trace]]],
    prices: dict[str, Price],
) -> str:
    """Cost per ticket and the share of runs that met the required actions, with which
    conditions are on the frontier: not beaten on both."""
    points = []
    rows = []
    for label, model, traces in conditions:
        n = len(traces)
        met = sum(outcome(cases[t.case_id], t) == "met" for t in traces)
        cost = sum(run_cost(t, prices[model]) for t in traces) / n
        low, high = wilson_interval(met, n)
        points.append((label, cost, met / n))
        rows.append((label, cost, met, n, low, high))
    front = set(pareto(points))
    lines = [
        f"{'condition':<19}{'$ per run':>10}{'met':>8}{'share':>7}{'95% interval':>14}"
        + f"{'frontier':>10}"
    ]
    for label, cost, met, n, low, high in rows:
        lines.append(
            f"{label:<19}{cost:>10.4f}{f'{met}/{n}':>8}{f'{100 * met / n:.0f}%':>7}"
            f"{f'{100 * low:.0f}-{100 * high:.0f}%':>14}"
            f"{'yes' if label in front else 'no':>10}"
        )
    return "\n".join(lines) + "\n"


def render_plan(
    conditions: list[tuple[str, str, list[Trace]]],
    prices: dict[str, Price],
    tickets: tuple[int, ...] = (88, 161, 466),
    gaps: tuple[int, ...] = (15, 10, 5),
) -> str:
    """What it costs to tell two versions apart: tickets from chapter 18's power plan,
    each run once under each version, at this condition's cost per run."""
    lines = [
        f"{'gap to detect':<19}" + "".join(f"{f'{g} pts':>10}" for g in gaps),
        f"{'tickets needed':<19}" + "".join(f"{n:>10}" for n in tickets),
    ]
    for label, model, traces in conditions:
        per_run = sum(run_cost(t, prices[model]) for t in traces) / len(traces)
        cells = "".join(f"{2 * n * per_run:>10.2f}" for n in tickets)
        lines.append(f"{label + ' ($)':<19}{cells}")
    return "\n".join(lines) + "\n"


def render_thinking(
    conditions: list[tuple[str, str, list[Trace]]], prices: dict[str, Price]
) -> str:
    """How much of what is billed the model never showed: per run, the tokens it wrote,
    the tokens it thought, the thinking share of output, and the cost counting only the
    written tokens against the cost of everything billed."""
    lines = [
        f"{'condition':<19}{'written':>9}{'thought':>9}{'share':>7}"
        + f"{'$ written':>11}{'$ billed':>10}"
    ]
    for label, model, traces in conditions:
        n = len(traces)
        thought = sum(t.usage.get("thinking_tokens", 0) for t in traces) / n
        out = sum(t.usage["output_tokens"] for t in traces) / n
        billed = sum(run_cost(t, prices[model]) for t in traces) / n
        unseen = thought * prices[model][1] / 1_000_000
        lines.append(
            f"{label:<19}{out - thought:>9.0f}{thought:>9.0f}"
            + f"{f'{100 * thought / out:.0f}%':>7}{billed - unseen:>11.4f}{billed:>10.4f}"
        )
    return "\n".join(lines) + "\n"


def _met_by_ticket(cases: dict[str, EvalCase], traces: list[Trace]) -> dict[str, int]:
    out: dict[str, int] = {}
    for t in traces:
        met = outcome(cases[t.case_id], t) == "met"
        out[t.case_id] = out.get(t.case_id, 0) + met
    return out


def render_paired(
    cases: dict[str, EvalCase],
    baseline: list[Trace],
    others: list[tuple[str, list[Trace]]],
    baseline_label: str,
) -> str:
    """Each other condition against the baseline, ticket by ticket. A ticket is better
    when more of its runs met the required actions under the other condition, worse when
    fewer, the same when equal. Then the exact sign test on the tickets that differ, and
    the difference in the share of runs that met, redrawing whole tickets."""
    base = _met_by_ticket(cases, baseline)
    trials = len(baseline) // len(base)
    lines = [
        f"{'against ' + baseline_label:<24}{'better':>7}{'worse':>7}{'same':>6}"
        + f"{'p':>7}{'difference, points':>26}"
    ]
    for label, traces in others:
        mine = _met_by_ticket(cases, traces)
        ids = sorted(set(base) & set(mine))
        better = sum(mine[i] > base[i] for i in ids)
        worse = sum(mine[i] < base[i] for i in ids)
        diffs = [(mine[i] - base[i]) / trials for i in ids]
        low, high = compare.cluster_interval(compare.mean, diffs)
        shown = (
            f"{100 * compare.mean(diffs):+.1f} ({100 * low:+.1f} to {100 * high:+.1f})"
        )
        p = compare.exact_sign_test(better, worse)
        lines.append(
            f"{label:<24}{better:>7}{worse:>7}{len(ids) - better - worse:>6}"
            + f"{'<0.001' if p < 0.001 else f'{p:.3f}':>7}{shown:>26}"
        )
    return "\n".join(lines) + "\n"


def render_repeat(
    cases: dict[str, EvalCase], pairs: list[tuple[str, list[Trace], list[Trace]]]
) -> str:
    """The same condition run twice, compared as if it were two versions. Anything this
    finds is noise, because nothing changed. Each pair is (label, first, second)."""
    rows = [
        render_paired(cases, first, [(label, second)], "first run").splitlines()
        for label, first, second in pairs
    ]
    return "\n".join([rows[0][0]] + [r[1] for r in rows]) + "\n"


def summary_row(cases: dict[str, EvalCase], traces: list[Trace], price: Price) -> Any:
    """Cost and outcomes of one condition as plain numbers, for tests and figures."""
    n = len(traces)
    met = sum(outcome(cases[t.case_id], t) == "met" for t in traces)
    total = sum(run_cost(t, price) for t in traces)
    return {"runs": n, "met": met, "dollars": total, "per_run": total / n}
