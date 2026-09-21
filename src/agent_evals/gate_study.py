"""How well does a regression gate tell a real drop from luck? (chapter 23)

The gate's rules are tried on recorded runs of the same 42 tickets. Each condition has
six recorded passes, a pass being one run of every ticket, and a gate looks at three
passes on each side, as a CI job that runs three trials would. Splitting one condition's
six passes into two threes gives pairs where nothing changed, so any block among them is
a false alarm. Pairing the passes of two different conditions gives pairs where the drop
is real, and its size is known from all the passes. Every split is tried and none is
sampled, so the tables are the same every time.
"""

import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from agent_evals import compare, regression, reliability
from agent_evals.schema import EvalCase, Trace

SIDE = 3

# (label, method, margin in points)
RULES = (
    ("point, drop over 5", "point", 5.0),
    ("sign test", "sign", 0.0),
    ("interval, margin 10", "interval", 10.0),
    ("interval, margin 5", "interval", 5.0),
)

SHORT = {
    "point, drop over 5": "point",
    "sign test": "sign",
    "interval, margin 10": "int 10",
    "interval, margin 5": "int 5",
}

# The recorded runs of chapter 22: two attempts of three trials for each condition.
CONDITIONS = {
    "3.6-flash": ("triage-cost-3-6", "triage-metered-3-6"),
    "3.6 + guard": ("triage-cost-3-6-guard", "triage-metered-3-6-guard"),
    "2.5-flash": ("triage-cost-2-5", "triage-metered-2-5"),
    "3.5-lite": ("triage-cost-3-5-lite", "triage-metered-3-5-lite"),
}

# (baseline, candidate): a drop for the candidate, or a gain when negative.
PAIRS = (
    ("3.6 + guard", "3.6-flash"),
    ("3.6 + guard", "2.5-flash"),
    ("2.5-flash", "3.6-flash"),
    ("2.5-flash", "3.5-lite"),
    ("3.5-lite", "3.6-flash"),
    ("3.6-flash", "2.5-flash"),
    ("3.6-flash", "3.6 + guard"),
)

Pass = dict[str, bool]


def passes(cases: dict[str, EvalCase], runs: list[list[Trace]]) -> list[Pass]:
    """Each trial of each run as one pass over the tickets: {ticket: met}."""
    out: list[Pass] = []
    for traces in runs:
        by_trial: dict[int, Pass] = {}
        for t in traces:
            by_trial.setdefault(t.trial, {})[t.case_id] = reliability.success(
                cases[t.case_id], t
            )
        out += [by_trial[k] for k in sorted(by_trial)]
    return out


def _diffs(base: list[Pass], cand: list[Pass]) -> list[float]:
    tickets = sorted(set(base[0]) & set(cand[0]))
    return [
        sum(p[k] for p in cand) / len(cand) - sum(p[k] for p in base) / len(base)
        for k in tickets
    ]


def measured_drop(base: list[Pass], cand: list[Pass]) -> float:
    """Points by which the candidate is worse over every recorded pass, negative when it
    is better."""
    diffs = _diffs(base, cand)
    return -100 * sum(diffs) / len(diffs)


@dataclass
class Outcome:
    block: float
    hold: float

    @property
    def passed(self) -> float:
        return 1 - self.block - self.hold


def _verdicts(diffs: list[float], seed: int) -> dict[str, str]:
    interval = compare.cluster_interval(compare.mean, diffs, resamples=1000, seed=seed)
    return {
        label: regression.decide(diffs, method, margin, interval=interval)[0]
        for label, method, margin in RULES
    }


def _tally(verdicts: list[str]) -> Outcome:
    n = len(verdicts)
    return Outcome(verdicts.count("block") / n, verdicts.count("inconclusive") / n)


def _triples(n: int) -> list[tuple[int, ...]]:
    return list(combinations(range(n), SIDE))


def nothing_changed(base: list[Pass]) -> dict[str, Outcome]:
    """Every way to split one condition's six passes into a baseline three and a
    candidate three: 20 ordered pairs where nothing changed."""
    votes: dict[str, list[str]] = {label: [] for label, _, _ in RULES}
    all_passes = set(range(len(base)))
    for i, s in enumerate(_triples(len(base))):
        rest = sorted(all_passes - set(s))
        diffs = _diffs([base[j] for j in s], [base[j] for j in rest])
        for label, verdict in _verdicts(diffs, i).items():
            votes[label].append(verdict)
    return {label: _tally(v) for label, v in votes.items()}


def real_drop(base: list[Pass], cand: list[Pass]) -> dict[str, Outcome]:
    """Every choice of three passes on each side: 400 pairs where the drop is real."""
    votes: dict[str, list[str]] = {label: [] for label, _, _ in RULES}
    i = 0
    for sb in _triples(len(base)):
        for sc in _triples(len(cand)):
            diffs = _diffs([base[j] for j in sb], [cand[j] for j in sc])
            for label, verdict in _verdicts(diffs, i).items():
                votes[label].append(verdict)
            i += 1
    return {label: _tally(v) for label, v in votes.items()}


def pooled(outcomes: list[dict[str, Outcome]]) -> dict[str, Outcome]:
    n = len(outcomes)
    return {
        label: Outcome(
            sum(o[label].block for o in outcomes) / n,
            sum(o[label].hold for o in outcomes) / n,
        )
        for label, _, _ in RULES
    }


def _pct(x: float) -> str:
    return f"{100 * x:.0f}"


def render_nothing_changed(outcomes: dict[str, Outcome], pairs: int) -> str:
    lines = [f"{'rule':<22}{'blocked':>9}{'held':>7}{'passed':>8}"]
    for label, _, _ in RULES:
        o = outcomes[label]
        lines.append(
            f"{label:<22}{_pct(o.block) + '%':>9}{_pct(o.hold) + '%':>7}"
            + f"{_pct(o.passed) + '%':>8}"
        )
    lines.append(f"{pairs} pairs of three passes where nothing changed")
    return "\n".join(lines) + "\n"


def render_drops(rows: list[tuple[str, float, dict[str, Outcome]]]) -> str:
    """One row per pair of conditions: the measured drop, then for each rule the share
    of draws blocked and the share held, as blocked/held."""
    head = f"{'baseline -> candidate':<26}{'drop':>6}"
    for label, _, _ in RULES:
        head += f"{SHORT[label]:>9}"
    lines = [head]
    for name, drop, outcomes in rows:
        cells = "".join(
            f"{_pct(outcomes[label].block) + '/' + _pct(outcomes[label].hold):>9}"
            for label, _, _ in RULES
        )
        lines.append(f"{name:<26}{drop:>+6.1f}{cells}")
    lines.append("drop in points, then blocked/held as a % of 400 draws")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------- retrying


def _rates(base: list[Pass]) -> dict[str, float]:
    return {k: sum(p[k] for p in base) / len(base) for k in base[0]}


def _draw(rng: random.Random, rates: dict[str, float]) -> Pass:
    return {k: rng.random() < r for k, r in rates.items()}


def retry_nothing_changed(base: list[Pass], draws: int, seed: int = 0):
    """A simulation, because six passes cannot make three sets of three. Each ticket
    keeps the rate it had over the recorded passes, and every simulated pass is drawn
    from those rates. Returns {rule: (blocked at first try, blocked after a retry)}."""
    rng = random.Random(seed)
    rates = _rates(base)
    first = {label: 0 for label, _, _ in RULES}
    both = dict(first)
    for i in range(draws):
        b = [_draw(rng, rates) for _ in range(SIDE)]
        v1 = _verdicts(_diffs(b, [_draw(rng, rates) for _ in range(SIDE)]), 2 * i)
        v2 = _verdicts(_diffs(b, [_draw(rng, rates) for _ in range(SIDE)]), 2 * i + 1)
        for label, _, _ in RULES:
            no1 = v1[label] != "pass"
            first[label] += no1
            both[label] += no1 and v2[label] != "pass"
    return {label: (first[label] / draws, both[label] / draws) for label in first}


def retry_real_drop(base: list[Pass], cand: list[Pass]):
    """The candidate's six passes split into two tries of three, and the baseline any
    three. Returns {rule: (not passed at first try, not passed at both tries)}."""
    first = {label: 0 for label, _, _ in RULES}
    both = dict(first)
    n = 0
    everything = set(range(len(cand)))
    for sb in _triples(len(base)):
        for sc in _triples(len(cand)):
            other = sorted(everything - set(sc))
            b = [base[j] for j in sb]
            v1 = _verdicts(_diffs(b, [cand[j] for j in sc]), n)
            v2 = _verdicts(_diffs(b, [cand[j] for j in other]), n + 100000)
            for label, _, _ in RULES:
                no1 = v1[label] != "pass"
                first[label] += no1
                both[label] += no1 and v2[label] != "pass"
            n += 1
    return {label: (first[label] / n, both[label] / n) for label in first}


def render_retry(
    scenarios: list[tuple[str, dict[str, tuple[float, float]]]],
) -> str:
    """Rows are rules and columns are scenarios, each cell one try / after a retry, as
    the share of draws that did not pass."""
    head = f"{'rule':<22}" + "".join(f"{name:>18}" for name, _ in scenarios)
    lines = [head]
    for label, _, _ in RULES:
        cells = "".join(
            f"{_pct(t[label][0]) + ' / ' + _pct(t[label][1]):>18}" for _, t in scenarios
        )
        lines.append(f"{label:<22}{cells}")
    lines.append("not passed at one try / not passed after a retry, %")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------- loading


def load(root: Path, cases: dict[str, EvalCase]) -> dict[str, list[Pass]]:
    return {
        name: passes(
            cases,
            [
                regression.read_traces(root / "runs" / run / "traces.jsonl")
                for run in runs
            ],
        )
        for name, runs in CONDITIONS.items()
    }
