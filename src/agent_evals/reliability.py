"""How reliable is an agent over repeated runs of the same case? (chapter 19)

A case is run several times. Each run either meets the case's required actions or it
does not. From the counts per case this module gives pass@k (at least one of k runs
succeeds), pass^k (all k succeed), how the cases split into always, never and
sometimes, how much of the variation is between cases and how much between runs of the
same case, and what that means for how many cases and runs to buy. Standard library.
"""

from collections.abc import Callable
from math import comb, sqrt

from agent_evals.compare import cluster_interval, mean
from agent_evals.graders import grade
from agent_evals.schema import EvalCase, Trace
from agent_evals.stats import wilson_interval

TRIAL_RUNS = (
    "triage-heldout-v1",
    "triage-heldout-v1-2",
    "triage-heldout-v1-3",
    "triage-reliability-5x",
)


def success(case: EvalCase, trace: Trace) -> bool:
    """The Chapter 7 measure: every required action was taken and the run did not
    end in an error."""
    return grade(case, trace).required_actions_met and trace.error is None


def outcomes(
    cases: dict[str, EvalCase], traces_by_run: list[list[Trace]]
) -> dict[str, list[bool]]:
    """{case: [success in each trial]}, trials in the order the runs are given."""
    out: dict[str, list[bool]] = {}
    for traces in traces_by_run:
        for t in traces:
            out.setdefault(t.case_id, []).append(success(cases[t.case_id], t))
    return out


def errors_by_case(traces_by_run: list[list[Trace]]) -> dict[str, list[bool]]:
    out: dict[str, list[bool]] = {}
    for traces in traces_by_run:
        for t in traces:
            out.setdefault(t.case_id, []).append(t.error is not None)
    return out


# ------------------------------------------------------------- pass@k, pass^k


def pass_hat(c: int, n: int, k: int) -> float:
    """The chance that k trials drawn from a case's n recorded trials, c of them
    successful, all succeed. This is the pass^k estimator of tau-bench."""
    return comb(c, k) / comb(n, k)


def pass_at(c: int, n: int, k: int) -> float:
    """The chance that at least one of k trials drawn from n recorded trials, c
    successful, succeeds. This is the unbiased pass@k estimator of the Codex paper."""
    return 1 - comb(n - c, k) / comb(n, k)


def per_case(
    outs: dict[str, list[bool]], k: int, estimator: Callable[[int, int, int], float]
) -> list[float]:
    return [estimator(sum(v), len(v), k) for v in outs.values()]


def curve(outs: dict[str, list[bool]], ks: range) -> list[tuple[int, float, float]]:
    return [
        (
            k,
            mean(per_case(outs, k, pass_at)),
            mean(per_case(outs, k, pass_hat)),
        )
        for k in ks
    ]


def split_counts(outs: dict[str, list[bool]]) -> tuple[int, int, int]:
    """Cases that always succeed, never succeed, and sometimes succeed."""
    always = sum(all(v) for v in outs.values())
    never = sum(not any(v) for v in outs.values())
    return always, never, len(outs) - always - never


# --------------------------------------------------------------- variance


def variance_components(
    outs: dict[str, list[bool]],
) -> tuple[float, float, float, float]:
    """One-way random-effects ANOVA on 0/1 outcomes with the same number of trials
    per case. Returns (mean square between cases, mean square within a case,
    variance between cases, variance within a case). The between-case variance is
    clipped at zero."""
    n = len(outs)
    k = len(next(iter(outs.values())))
    if any(len(v) != k for v in outs.values()) or k < 2 or n < 2:
        raise ValueError("needs the same number of trials, at least 2, for 2+ cases")
    means = [sum(v) / k for v in outs.values()]
    grand = sum(means) / n
    msb = k * sum((m - grand) ** 2 for m in means) / (n - 1)
    msw = sum((x - m) ** 2 for v, m in zip(outs.values(), means) for x in v) / (
        n * (k - 1)
    )
    return msb, msw, max(0.0, (msb - msw) / k), msw


def standard_error(var_between: float, var_within: float, cases: int, trials: int):
    """Standard error of the mean success rate with this many cases and trials each."""
    return sqrt((var_between + var_within / trials) / cases)


def chance_of_looking_consistent(p: float, trials: int) -> float:
    """A case that truly succeeds with chance p, run `trials` times: how often all the
    trials agree, so that it looks like an always or a never."""
    return p**trials + (1 - p) ** trials


# ---------------------------------------------------------------- the reports


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _span(interval: tuple[float, float]) -> str:
    return f"{100 * interval[0]:.0f}-{100 * interval[1]:.0f}%"


def render_trials(cases, names, traces_by_run) -> str:
    """Each recorded run on its own: how many cases met the required actions."""
    lines = [f"{'run':<26}{'cases met':>12}{'95% interval':>16}{'errors':>9}"]
    for name, traces in zip(names, traces_by_run):
        by_trial: dict[int, list[Trace]] = {}
        for t in traces:
            by_trial.setdefault(t.trial, []).append(t)
        for trial, group in sorted(by_trial.items()):
            k = sum(success(cases[t.case_id], t) for t in group)
            label = name if len(by_trial) == 1 else f"{name} trial {trial}"
            errs = sum(t.error is not None for t in group)
            lines.append(
                f"{label:<26}{f'{k} of {len(group)}':>12}"
                f"{_span(wilson_interval(k, len(group))):>16}{errs:>9}"
            )
    return "\n".join(lines) + "\n"


def render_curve(outs: dict[str, list[bool]], top: int | None = None) -> str:
    n = len(next(iter(outs.values())))
    ks = range(1, (top or n) + 1)
    p1 = mean(per_case(outs, 1, pass_hat))
    lines = [
        f"{len(outs)} cases, {n} trials each",
        f"{'k':>3}{'pass@k':>9}{'pass^k':>9}{'95% interval':>15}{'if independent':>17}",
    ]
    for k, at, hat in curve(outs, ks):
        vector = per_case(outs, k, pass_hat)
        span = _span(cluster_interval(mean, vector))
        lines.append(f"{k:>3}{_pct(at):>9}{_pct(hat):>9}{span:>15}{_pct(p1**k):>17}")
    return "\n".join(lines) + "\n"


def render_split(outs: dict[str, list[bool]]) -> str:
    always, never, sometimes = split_counts(outs)
    n = len(next(iter(outs.values())))
    lines = [
        f"{'over ' + str(n) + ' trials':<24}{'cases':>7}",
        f"{'succeeded every time':<24}{always:>7}",
        f"{'never succeeded':<24}{never:>7}",
        f"{'sometimes':<24}{sometimes:>7}",
        "",
        f"{'successes in n trials':<24}{'cases':>7}",
    ]
    counts: dict[int, int] = {}
    for v in outs.values():
        counts[sum(v)] = counts.get(sum(v), 0) + 1
    for c in range(n + 1):
        lines.append(f"{f'{c} of {n}':<24}{counts.get(c, 0):>7}")
    return "\n".join(lines) + "\n"


def render_variance(outs: dict[str, list[bool]]) -> str:
    _, _, var_b, var_w = variance_components(outs)
    trials = len(next(iter(outs.values())))
    icc = var_b / (var_b + var_w) if var_b + var_w else 0.0
    lines = [
        f"between cases    {var_b:.3f}",
        f"within a case    {var_w:.3f}",
        f"share of the spread that is between cases: {_pct(icc)}",
        "",
        "standard error of the success rate, in points",
        f"{'cases':<8}" + "".join(f"{f'{t} trials':>12}" for t in (1, 3, 8, 1000)),
    ]
    for cases in (len(outs), 2 * len(outs), 4 * len(outs)):
        cells = "".join(
            f"{100 * standard_error(var_b, var_w, cases, t):>12.1f}"
            for t in (1, 3, 8, 1000)
        )
        lines.append(f"{cases:<8}{cells}")
    lines.append(f"(estimated from {len(outs)} cases with {trials} trials each)")
    return "\n".join(lines) + "\n"


def render_consistent(trials=(3, 8), rates=(0.5, 0.7, 0.9)) -> str:
    lines = [
        f"{'true success chance':<22}" + "".join(f"{f'{t} trials':>10}" for t in trials)
    ]
    for p in rates:
        cells = "".join(
            f"{_pct(chance_of_looking_consistent(p, t)):>10}" for t in trials
        )
        lines.append(f"{_pct(p):<22}{cells}")
    return "\n".join(lines) + "\n"


def render_errors(errs: dict[str, list[bool]]) -> str:
    n = len(next(iter(errs.values())))
    counts: dict[int, int] = {}
    for v in errs.values():
        counts[sum(v)] = counts.get(sum(v), 0) + 1
    lines = [f"{'runs that ended in an error':<30}{'cases':>7}"]
    for c in range(n + 1):
        lines.append(f"{f'{c} of {n}':<30}{counts.get(c, 0):>7}")
    return "\n".join(lines) + "\n"
