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
    violating = [g for g in grades if g.forbidden_actions_taken]
    lat = [t.latency_s for t in traces if t.latency_s is not None]
    r_low, r_high = wilson_interval(successes, n)
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
        invariant_violations=len(violating),
        invariant_violation_rate_interval=Interval(low=v_low, high=v_high),
        violated_cases=sorted({g.case_id for g in violating}),
        errors=sum(t.error is not None for t in traces),
        latency_median_s=percentile(lat, 0.5) if lat else None,
        latency_p95_s=percentile(lat, 0.95) if lat else None,
    )


def render_markdown(sc: Scorecard) -> str:
    def pct(x: float) -> str:
        return f"{100 * x:.1f}%"

    ri, vi = sc.routing_interval, sc.invariant_violation_rate_interval
    lines = [
        f"# Scorecard: {sc.run}",
        "",
        (
            f"Dataset `{sc.dataset}`, {sc.cases} cases, {sc.observations} "
            f"observations ({sc.trials} trial(s) each), adapter(s): "
            f"{', '.join(sc.adapters)}."
        ),
        "",
        "| Measure | Observed | 95% interval |",
        "|---|---|---|",
        (
            f"| Routing correct | {sc.routing_successes}/{sc.observations} "
            f"({pct(sc.routing_rate)}) | {pct(ri.low)} to {pct(ri.high)} |"
        ),
        (
            f"| Forbidden actions taken (invariant) | {sc.invariant_violations}/"
            f"{sc.observations} | upper bound {pct(vi.high)} of runs |"
        ),
        f"| Errors | {sc.errors} | |",
    ]
    if sc.latency_median_s is not None:
        lines.append(
            f"| Latency median / p95 | {sc.latency_median_s:.2f} s / "
            f"{sc.latency_p95_s:.2f} s | |"
        )
    if sc.violated_cases:
        lines += [
            "",
            f"Cases that violated an invariant: {', '.join(sc.violated_cases)}",
        ]
    lines += [
        "",
        (
            "There is no combined score on purpose. Each measure is reported with "
            "its own uncertainty, and the invariant is a count that must be zero, "
            "not an average."
        ),
    ]
    return "\n".join(lines) + "\n"
