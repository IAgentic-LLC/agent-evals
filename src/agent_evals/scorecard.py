"""Build a scorecard (a profile of measures, each with its uncertainty) from traces."""

from agent_evals.graders import grade
from agent_evals.schema import EvalCase, Interval, Scorecard, Trace
from agent_evals.stats import percentile, wilson_interval


def build_scorecard(
    dataset: str, run: str, cases: list[EvalCase], traces: list[Trace]
) -> Scorecard:
    by_id = {c.case_id: c for c in cases}
    grades = [grade(by_id[t.case_id], t) for t in traces]
    n = len(grades)
    successes = sum(g.routing_correct for g in grades)
    actions_ok = sum(
        g.required_actions_met and t.error is None for g, t in zip(grades, traces)
    )
    violating = [g for g in grades if g.forbidden_actions_taken]
    lat = [t.latency_s for t in traces if t.latency_s is not None]
    r_low, r_high = wilson_interval(successes, n)
    a_low, a_high = wilson_interval(actions_ok, n)
    v_low, v_high = wilson_interval(len(violating), n)
    return Scorecard(
        dataset=dataset,
        run=run,
        adapters=sorted({t.adapter for t in traces}),
        cases=len(cases),
        trials=max((t.trial for t in traces), default=1),
        observations=n,
        routing_successes=successes,
        routing_rate=successes / n,
        routing_interval=Interval(low=r_low, high=r_high),
        actions_successes=actions_ok,
        actions_rate=actions_ok / n,
        actions_interval=Interval(low=a_low, high=a_high),
        invariant_violations=len(violating),
        invariant_violation_rate_interval=Interval(low=v_low, high=v_high),
        violated_cases=sorted({g.case_id for g in violating}),
        errors=sum(t.error is not None for t in traces),
        latency_median_s=percentile(lat, 0.5) if lat else None,
        latency_p95_s=percentile(lat, 0.95) if lat else None,
    )


def _table(rows: list[tuple[str, str, str]]) -> list[str]:
    """A markdown table with padded columns, so it also reads well as plain text."""
    header = ("Measure", "Observed", "95% interval")
    body = [header, *rows]
    widths = [max(len(r[i]) for r in body) for i in range(3)]

    def fmt(row: tuple[str, str, str]) -> str:
        return "| " + " | ".join(c.ljust(w) for c, w in zip(row, widths)) + " |"

    sep = "|" + "|".join("-" * (w + 2) for w in widths) + "|"
    return [fmt(header), sep, *[fmt(r) for r in rows]]


def render_markdown(sc: Scorecard) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    ri, ai = sc.routing_interval, sc.actions_interval
    vi = sc.invariant_violation_rate_interval
    n = sc.observations
    rows = [
        (
            "Routing correct",
            f"{sc.routing_successes}/{n} ({pct(sc.routing_rate)})",
            f"{pct(ri.low)} to {pct(ri.high)}",
        ),
        (
            "Required actions taken",
            f"{sc.actions_successes}/{n} ({pct(sc.actions_rate)})",
            f"{pct(ai.low)} to {pct(ai.high)}",
        ),
        (
            "Forbidden actions taken",
            f"{sc.invariant_violations}/{n}",
            f"upper bound {pct(vi.high)} of runs",
        ),
        ("Errors", str(sc.errors), ""),
    ]
    if sc.latency_median_s is not None:
        rows.append(
            (
                "Latency median / p95",
                f"{sc.latency_median_s:.2f} s / {sc.latency_p95_s:.2f} s",
                "",
            )
        )
    lines = [
        f"# Scorecard: {sc.run}",
        "",
        f"Dataset `{sc.dataset}`: {sc.cases} cases, {n} observations",
        f"({sc.trials} trial(s) each). Adapter: {', '.join(sc.adapters)}.",
        "",
        *_table(rows),
    ]
    if sc.violated_cases:
        lines += ["", f"Invariant violated in: {', '.join(sc.violated_cases)}"]
    lines += [
        "",
        "There is no combined score on purpose. Each measure carries its own",
        "uncertainty, and the invariant is a count that must be zero, not an",
        "average.",
    ]
    return "\n".join(lines) + "\n"
