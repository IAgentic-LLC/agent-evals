"""Watching a deployed product: what can be measured with no labels, and what a monitor
can and cannot tell you (chapter 24).

There is no live traffic behind this chapter. Every number comes from recorded runs. A
"day" of traffic is drawn from them: pick tickets at random, and for each one pick one of
the outcomes that ticket had in the recorded passes. That gives a stream whose true rate
is known, so a monitor can be tested against a fall whose size and start are known, and
against a stream where nothing changed.

The one measure used online is the share of runs that ended without an error. It needs no
expected answer, so it can be computed for a ticket nobody has labeled. It is not the met
rate of the earlier chapters, and `render_proxy` says how far apart they are.
"""

import random
from collections import Counter
from dataclasses import dataclass
from math import log2, sqrt
from pathlib import Path

from agent_evals import compare, gate_study, reliability
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

MONITORS = ("threshold", "shewhart", "cusum")
BASELINE_DAYS = 14
MONITOR_DAYS = 30
QUIET_DAYS = 5  # ordinary days between the baseline and the change
SHIFT_SIGMA = 0.5  # CUSUM allowance, half the shift it is tuned to catch
CUSUM_LIMIT = 4.0


@dataclass
class Population:
    """A stream of tickets with known outcomes: per ticket, its slice value and the
    error-free outcome of each recorded pass."""

    tickets: list[tuple[str, list[bool]]]

    def day(self, rng: random.Random, n: int) -> list[tuple[str, bool]]:
        """n tickets drawn at random, each with one of its recorded outcomes."""
        out = []
        for _ in range(n):
            stratum, outcomes = self.tickets[rng.randrange(len(self.tickets))]
            out.append((stratum, outcomes[rng.randrange(len(outcomes))]))
        return out

    @property
    def rate(self) -> float:
        flat = [o for _, outcomes in self.tickets for o in outcomes]
        return sum(flat) / len(flat)


def population(
    cases: dict[str, EvalCase], runs: list[list[Trace]], key: str = "specialist"
) -> Population:
    """Every recorded run of every ticket, grouped by ticket."""
    by_ticket: dict[str, list[bool]] = {}
    for traces in runs:
        for t in traces:
            by_ticket.setdefault(t.case_id, []).append(t.error is None)
    return Population(
        [(cases[k].slices.get(key, "(none)"), v) for k, v in sorted(by_ticket.items())]
    )


def error_free(traces: list[Trace]) -> float:
    return sum(t.error is None for t in traces) / len(traces)


# ---------------------------------------------------------------------- monitors


def _p0(baseline: list[tuple[int, int]]) -> float:
    k = sum(x for x, _ in baseline)
    n = sum(m for _, m in baseline)
    return min(0.98, max(0.02, k / n))


def alarm_days(
    monitor: str, baseline: list[tuple[int, int]], watch: list[tuple[int, int]]
) -> list[int]:
    """Every day (counting from 1) on which the monitor alarms. The baseline days set the
    level. Each day is (runs without an error, runs). A CUSUM starts again from zero
    after it alarms, as a person would reset it after looking."""
    p0 = _p0(baseline)
    s = 0.0
    days = []
    for day, (k, n) in enumerate(watch, start=1):
        if n == 0:
            continue
        p = k / n
        sigma = sqrt(p0 * (1 - p0) / n)
        if monitor == "threshold":
            hit = p < p0 - 0.10
        elif monitor == "shewhart":
            hit = p < p0 - 3 * sigma
        else:
            s = max(0.0, s - (p - p0) / sigma - SHIFT_SIGMA)
            hit = s > CUSUM_LIMIT
            if hit:
                s = 0.0
        if hit:
            days.append(day)
    return days


def first_alarm(
    monitor: str, baseline: list[tuple[int, int]], watch: list[tuple[int, int]]
) -> int | None:
    """The first day (counting from 1) on which the monitor alarms, or None."""
    days = alarm_days(monitor, baseline, watch)
    return days[0] if days else None


def _observe(
    pop: Population, rng: random.Random, n: int, stratum: str | None
) -> tuple[int, int]:
    rows = pop.day(rng, n)
    if stratum is not None:
        rows = [r for r in rows if r[0] == stratum]
    return sum(ok for _, ok in rows), len(rows)


def stream(
    schedule: list[Population], rng: random.Random, n: int, stratum: str | None = None
) -> list[tuple[int, int]]:
    """One (runs without an error, runs) pair for each day of the schedule. With a
    stratum, only the tickets of that slice are counted."""
    return [_observe(pop, rng, n, stratum) for pop in schedule]


def false_alarms(
    pop: Population, n: int, draws: int, seed: int = 0, stratum: str | None = None
) -> dict[str, float]:
    """The share of streams, in which nothing changed, that raise an alarm within the
    monitored days."""
    rng = random.Random(seed)
    hits = {m: 0 for m in MONITORS}
    for _ in range(draws):
        obs = stream([pop] * (BASELINE_DAYS + MONITOR_DAYS), rng, n, stratum)
        base, watch = obs[:BASELINE_DAYS], obs[BASELINE_DAYS:]
        for m in MONITORS:
            hits[m] += first_alarm(m, base, watch) is not None
    return {m: hits[m] / draws for m in MONITORS}


def detection(
    before: Population,
    after: Population,
    n: int,
    draws: int,
    seed: int = 0,
    stratum: str | None = None,
) -> dict[str, list[int | None]]:
    """For each monitor, the day of the first alarm on or after the day the stream
    changed, counted from that day (1 is the day it changed, None if it never alarmed),
    one entry per draw. The stream has the baseline days, then QUIET_DAYS more ordinary
    days, then the change. Alarms in the ordinary days are false alarms. They are not
    counted here, and `false_alarms` counts them."""
    rng = random.Random(seed)
    out: dict[str, list[int | None]] = {m: [] for m in MONITORS}
    for _ in range(draws):
        schedule = [before] * (BASELINE_DAYS + QUIET_DAYS) + [after] * MONITOR_DAYS
        obs = stream(schedule, rng, n, stratum)
        base = obs[:BASELINE_DAYS]
        for m in MONITORS:
            later = [
                d - QUIET_DAYS
                for d in alarm_days(m, base, obs[BASELINE_DAYS:])
                if d > QUIET_DAYS
            ]
            out[m].append(later[0] if later else None)
    return out


def _within(days: list[int | None], limit: int) -> float:
    return sum(d is not None and d <= limit for d in days) / len(days)


def _median(days: list[int | None], limit: int) -> str:
    """The median day among the streams that alarmed within the limit."""
    found = sorted(d for d in days if d is not None and d <= limit)
    return f"{found[len(found) // 2]}" if found else "-"


def render_false_alarms(rows: list[tuple[int, dict[str, float]]], days: int) -> str:
    lines = [f"{'tickets a day':<15}" + "".join(f"{m:>11}" for m in MONITORS)]
    for n, rates in rows:
        cells = "".join(f"{f'{100 * rates[m]:.0f}%':>11}" for m in MONITORS)
        lines.append(f"{n:<15}{cells}")
    lines.append(f"share of {days}-day streams with no change that raised an alarm")
    return "\n".join(lines) + "\n"


def render_detection(
    rows: list[tuple[str, int, dict[str, list[int | None]]]],
    within: int,
    what: str = "fall",
) -> str:
    """One row per change and traffic level: for each monitor, the share of streams that
    alarmed within `within` days of the change, and the median day."""
    head = f"{what:<22}{'tickets':>8}"
    for m in MONITORS:
        head += f"{m:>14}"
    lines = [head]
    for name, n, found in rows:
        cells = ""
        for m in MONITORS:
            pct = 100 * _within(found[m], within)
            share = "<1%" if 0 < pct < 0.5 else f"{pct:.0f}%"
            cells += f"{f'{share} day {_median(found[m], within)}':>14}"
        lines.append(f"{name:<22}{n:>8}{cells}")
    lines.append(f"alarmed within {within} days of the change, then the median day")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- the proxy


def render_proxy(
    rows: list[tuple[str, dict[str, EvalCase], list[Trace]]],
) -> str:
    """The online measure beside the offline one: runs without an error, runs that met
    the required actions, and the runs that were error-free and still missed."""
    lines = [f"{'run':<28}{'runs':>5}{'no error':>10}{'met':>7}{'gap':>6}{'missed':>8}"]
    for name, cases, traces in rows:
        n = len(traces)
        ok = sum(t.error is None for t in traces)
        met = sum(reliability.success(cases[t.case_id], t) for t in traces)
        lines.append(
            f"{name:<28}{n:>5}{f'{100 * ok / n:.1f}%':>10}{f'{100 * met / n:.1f}%':>7}"
            + f"{100 * (ok - met) / n:>6.1f}{ok - met:>8}"
        )
    lines.append(
        "gap is no error minus met, in points; missed is error-free but not met"
    )
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- the mix


def js_distance(p: dict[str, float], q: dict[str, float]) -> float:
    """The Jensen-Shannon distance between two shares, from 0 (the same) to 1."""
    keys = set(p) | set(q)
    m = {k: (p.get(k, 0) + q.get(k, 0)) / 2 for k in keys}

    def kl(a: dict[str, float]) -> float:
        return sum(a[k] * log2(a[k] / m[k]) for k in keys if a.get(k, 0) > 0)

    return sqrt((kl(p) + kl(q)) / 2)


def _shares(counts: Counter) -> dict[str, float]:
    total = sum(counts.values())
    return {k: v / total for k, v in counts.items()}


def render_mix(
    cases_a: dict[str, EvalCase],
    traces_a: list[Trace],
    cases_b: dict[str, EvalCase],
    traces_b: list[Trace],
    keys: tuple[str, str] = ("specialist", "runbook_topic"),
    names: tuple[str, str] = ("held-out", "change set"),
) -> str:
    """The same product on two sets of tickets: the met share overall and by slice, the
    distance between the mixes, and what the first set's rates give on the second's mix."""
    lines = []
    rate_a: dict[str, tuple[int, float]] = {}
    for key in keys:
        lines.append(f"{key:<16}{names[0]:>14}{names[1]:>14}")
        table = []
        for cases, traces in ((cases_a, traces_a), (cases_b, traces_b)):
            groups: dict[str, list[bool]] = {}
            for t in traces:
                groups.setdefault(
                    cases[t.case_id].slices.get(key, "(none)"), []
                ).append(reliability.success(cases[t.case_id], t))
            table.append(groups)
        if key == keys[-1]:
            rate_a = {k: (len(v), sum(v) / len(v)) for k, v in table[0].items()}
        for value in sorted(set(table[0]) | set(table[1])):
            cells = []
            for groups in table:
                g = groups.get(value)
                cells.append(f"{100 * sum(g) / len(g):.1f}% ({len(g)})" if g else "-")
            lines.append(f"{value:<16}{cells[0]:>14}{cells[1]:>14}")
        lines.append("")
    overall = [
        (sum(reliability.success(c[t.case_id], t) for t in tr), len(tr))
        for c, tr in ((cases_a, traces_a), (cases_b, traces_b))
    ]
    lines.append(
        f"{'all':<16}"
        + "".join(f"{f'{100 * k / n:.1f}% ({n})':>14}" for k, n in overall)
    )
    mix = [
        _shares(Counter(cases[t.case_id].slices.get(keys[0], "(none)") for t in traces))
        for cases, traces in ((cases_a, traces_a), (cases_b, traces_b))
    ]
    topic_b = Counter(
        cases_b[t.case_id].slices.get(keys[-1], "(none)") for t in traces_b
    )
    weights = _shares(topic_b)
    expected = sum(w * rate_a[k][1] for k, w in weights.items() if k in rate_a)
    lines.append("")
    lines.append(f"distance between the {keys[0]} mixes: {js_distance(*mix):.2f}")
    lines.append(
        f"{names[0]} rates on the {names[1]} {keys[-1]} mix: {100 * expected:.1f}%"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- shadowing


def shadow_power(
    live: Population,
    shadow: Population,
    n: int,
    draws: int,
    seed: int = 0,
    alpha: float = 0.05,
) -> float:
    """How often n shadowed tickets, each run under both versions, show the shadow
    version better on the error-free measure, by the exact sign test."""
    rng = random.Random(seed)
    won = 0
    for _ in range(draws):
        better = worse = 0
        for _ in range(n):
            i = rng.randrange(len(live.tickets))
            a = rng.choice(live.tickets[i][1])
            b = rng.choice(shadow.tickets[i][1])
            better += b and not a
            worse += a and not b
        if better > worse and compare.exact_sign_test(better, worse) < alpha:
            won += 1
    return won / draws


def render_shadow(rows: list[tuple[int, list[float]]], names: list[str]) -> str:
    lines = [f"{'tickets shadowed':<18}" + "".join(f"{n:>14}" for n in names)]
    for n, powers in rows:
        lines.append(f"{n:<18}" + "".join(f"{f'{100 * p:.0f}%':>14}" for p in powers))
    lines.append("share of draws in which the sign test showed the shadow was better")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------- review


def render_review(
    sizes: tuple[int, ...] = (25, 50, 100, 200, 400), rate: float = 0.29
) -> str:
    """What a sample of reviewed traces can say: the width of the interval on a common
    problem, and the chance of seeing a rare one at least once."""
    lines = [
        f"{'traces reviewed':<17}{f'{100 * rate:.0f}% problem':>16}{'1 in 100':>10}"
        + f"{'1 in 500':>10}"
    ]
    for n in sizes:
        low, high = wilson_interval(round(rate * n), n)
        seen = [1 - (1 - p) ** n for p in (0.01, 0.002)]
        lines.append(
            f"{n:<17}{f'{100 * low:.0f} to {100 * high:.0f}%':>16}"
            + f"{f'{100 * seen[0]:.0f}%':>10}{f'{100 * seen[1]:.0f}%':>10}"
        )
    lines.append(
        "interval on the common problem; chance of seeing the rare one at least once"
    )
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- loading


def load_defaults(root: Path) -> dict[str, Population]:
    """Populations for the study: the held-out tickets under the default and the guard
    and two cheaper models, the default's twelve passes, and the change set with and
    without the hint."""
    held = {c.case_id: c for c in load_cases(root / "datasets/triage_heldout_v1.jsonl")}
    chg = {c.case_id: c for c in load_cases(root / "datasets/triage_change_v1.jsonl")}

    def runs(names) -> list[list[Trace]]:
        return [read_traces(root / "runs" / n / "traces.jsonl") for n in names]

    out = {
        name: population(held, runs(names))
        for name, names in gate_study.CONDITIONS.items()
    }
    out["3.6-flash (12)"] = population(
        held, runs([*gate_study.CONDITIONS["3.6-flash"], *gate_study.EXTRA_RUNS])
    )
    out["hint"] = population(held, runs(["triage-gate-topics-heldout"]))
    out["change base"] = population(chg, runs(["triage-change-base"]))
    out["change hint"] = population(chg, runs(["triage-change-topics"]))
    default_runs = runs([*gate_study.CONDITIONS["3.6-flash"], *gate_study.EXTRA_RUNS])
    out["default by topic"] = population(held, default_runs, key="runbook_topic")
    out["change base by topic"] = population(
        chg, runs(["triage-change-base"]), key="runbook_topic"
    )
    return out
