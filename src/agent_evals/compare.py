"""Is one run really different from another? (chapter 18)

Everything here works on outcomes for the same questions under two conditions. The
unit of evidence is the question, not the answer: answers to one question, from
repeated runs, are not independent. Standard-library only.
"""

import random
from collections.abc import Callable
from functools import cache
from math import comb, exp, lgamma, log, sqrt
from statistics import NormalDist

from agent_evals import abstention
from agent_evals.judge_report import load_rows
from agent_evals.stats import wilson_interval

Z = NormalDist().inv_cdf(0.975)


# ------------------------------------------------------------------- tests


def exact_sign_test(only_a: int, only_b: int) -> float:
    """Two-sided exact p-value that two conditions differ, from the questions where
    they differ. It is McNemar's exact test: under no difference, each differing
    question is a fair coin. Questions where both agree carry no information."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_counts(a: list[bool], b: list[bool]) -> tuple[int, int, int, int]:
    """(both, only a, only b, neither) for two parallel lists of yes/no outcomes."""
    both = sum(x and y for x, y in zip(a, b))
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    return both, only_a, only_b, len(a) - both - only_a - only_b


def holm(pvalues: list[float]) -> list[float]:
    """Holm's step-down adjustment: controls the chance of any false alarm."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    out = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvalues[i])
        out[i] = min(1.0, running)
    return out


def benjamini_hochberg(pvalues: list[float]) -> list[float]:
    """Benjamini-Hochberg adjustment: controls the share of false alarms among
    the ones you report."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i], reverse=True)
    out = [0.0] * m
    running = 1.0
    for rank, i in enumerate(order):
        running = min(running, pvalues[i] * m / (m - rank))
        out[i] = running
    return out


# -------------------------------------------------------------- intervals


def newcombe_difference(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Newcombe's hybrid-score interval for p1 - p2 from two INDEPENDENT groups.
    It is what you get by treating two runs as unrelated samples."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson_interval(k1, n1)
    l2, u2 = wilson_interval(k2, n2)
    diff = p1 - p2
    return (
        diff - sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
        diff + sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2),
    )


def cluster_interval(
    stat: Callable[[list[float]], float],
    values: list[float],
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% percentile interval for a statistic of per-question values, redrawing
    whole questions with replacement."""
    rng = random.Random(seed)
    n = len(values)
    draws = sorted(
        stat([values[rng.randrange(n)] for _ in range(n)]) for _ in range(resamples)
    )
    return draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def wald_interval(k: int, n: int) -> tuple[float, float]:
    """The plain normal-approximation interval, the one to avoid at small n."""
    p = k / n
    half = Z * sqrt(p * (1 - p) / n)
    return p - half, p + half


def coverage(interval, n: int, p: float) -> float:
    """The exact chance that an interval built from n yes/no outcomes with true rate
    p contains p. A 95% interval should give 0.95. Sums over every possible count."""
    total = 0.0
    for k in range(n + 1):
        low, high = interval(k, n)
        if low <= p <= high:
            total += comb(n, k) * p**k * (1 - p) ** (n - k)
    return total


# ------------------------------------------------------------------- power


def _log_pmf(n: int, k: int, p: float) -> float:
    if p <= 0:
        return 0.0 if k == 0 else float("-inf")
    if p >= 1:
        return 0.0 if k == n else float("-inf")
    return (
        lgamma(n + 1)
        - lgamma(k + 1)
        - lgamma(n - k + 1)
        + k * log(p)
        + (n - k) * log(1 - p)
    )


def power_sign_test(n: int, delta: float, discordance: float, alpha: float = 0.05):
    """The chance the exact sign test finds a difference, exactly, with no simulation.

    n questions. `discordance` is the share of questions where the two conditions
    differ. `delta` is the true gap between the two rates: of the differing
    questions, the first condition wins (discordance + delta) / 2 of all questions.
    """
    p_first = (discordance + delta) / 2
    if not 0 <= p_first <= discordance or discordance > 1:
        raise ValueError("delta cannot exceed the discordance")
    lean = p_first / discordance if discordance else 0.5
    power = 0.0
    for m in range(1, n + 1):
        weight = exp(_log_pmf(n, m, discordance))
        if weight < 1e-15:
            continue
        low = _rejects_up_to(m, alpha)
        if low < 0:
            continue
        wins = [*range(low + 1), *range(m - low, m + 1)]
        power += weight * sum(exp(_log_pmf(m, b, lean)) for b in wins)
    return power


@cache
def _rejects_up_to(m: int, alpha: float) -> int:
    """The largest count i such that i (or fewer) wins out of m differing questions
    is already enough for the exact sign test to reject; -1 if none is."""
    best, tail = -1, 0
    for i in range(m // 2 + 1):
        tail += comb(m, i)
        if min(1.0, 2 * tail / 2**m) < alpha:
            best = i
        else:
            break
    return best


def questions_needed(
    delta: float, discordance: float, target: float = 0.8, limit: int = 3000
) -> int | None:
    """The first number of questions at which the sign test reaches the target
    power, or None. Exact tests have jagged power, so a few larger numbers can dip
    just below the target again."""
    for n in range(5, limit + 1):
        if power_sign_test(n, delta, discordance) >= target:
            return n
    return None


# ------------------------------------------------------------- the reports


def _pct(x: float) -> str:
    return f"{100 * x:.0f}%"


def _span(interval: tuple[float, float]) -> str:
    return f"{100 * interval[0]:.0f}-{100 * interval[1]:.0f}%"


def _points(interval: tuple[float, float]) -> str:
    return f"{100 * interval[0]:.1f} to {100 * interval[1]:.1f}"


def question_outcomes(cases, runs) -> dict[str, dict[str, list[bool]]]:
    """{prompt: {question: [refused in each run]}}, for the questions that should be
    answered. Which questions those are comes from the retrieval, which is the same
    in every run; a test checks that."""
    out: dict[str, dict[str, list[bool]]] = {}
    for name, traces_by_run in runs.items():
        per_question: dict[str, list[bool]] = {}
        for traces in traces_by_run:
            for t in traces:
                if abstention.should_answer(cases[t.case_id], t):
                    per_question.setdefault(t.case_id, []).append(
                        not abstention.answered(t)
                    )
        out[name] = per_question
    return out


def _rates(outcomes):
    """Per prompt, the refusal rate of each question (0, 0.5 or 1 for two runs)."""
    names = list(outcomes)
    questions = sorted(outcomes[names[0]])
    rates = {n: [mean(outcomes[n][q]) for q in questions] for n in names}
    return names, questions, rates


def _who_more(a: list[float], b: list[float]) -> tuple[int, int]:
    """Questions where the first refused more often, and where the second did."""
    return sum(x > y for x, y in zip(a, b)), sum(y > x for x, y in zip(a, b))


def render_intervals(cases, runs) -> str:
    outcomes = question_outcomes(cases, runs)
    names, questions, rates = _rates(outcomes)
    lines = [
        f"{'refused, should answer':<21}{'of':>7}{'Wald':>10}{'Wilson':>10}"
        + f"{'redrawing questions':>22}"
    ]
    for name in names:
        flat = [x for q in questions for x in outcomes[name][q]]
        k, n = sum(flat), len(flat)
        lines.append(
            f"{name:<21}{f'{k}/{n}':>7}{_span(wald_interval(k, n)):>10}"
            f"{_span(wilson_interval(k, n)):>10}"
            f"{_span(cluster_interval(mean, rates[name])):>22}"
        )
    return "\n".join(lines) + "\n"


def render_difference(cases, runs) -> str:
    outcomes = question_outcomes(cases, runs)
    names, questions, rates = _rates(outcomes)
    first, second = names
    flat = {n: [x for q in questions for x in outcomes[n][q]] for n in names}
    k1, k2 = sum(flat[first]), sum(flat[second])
    n1, n2 = len(flat[first]), len(flat[second])
    diff = [y - x for x, y in zip(rates[first], rates[second])]
    lines = [
        f"{second} minus {first}, refusals",
        f"{'':<40}{'points':>7}{'95% interval':>20}",
        (
            f"{'as two unrelated groups of answers':<40}"
            f"{100 * (k2 / n2 - k1 / n1):>7.1f}"
            f"{_points(newcombe_difference(k2, n2, k1, n1)):>20}"
        ),
        (
            f"{'same questions, redrawn together':<40}"
            f"{100 * mean(diff):>7.1f}{_points(cluster_interval(mean, diff)):>20}"
        ),
    ]
    for number in (0, 1):
        a = [outcomes[first][q][number] for q in questions]
        b = [outcomes[second][q][number] for q in questions]
        both, only_first_run, only_second_run, neither = paired_counts(a, b)
        low, high = bayes_paired_interval(
            only_second_run, only_first_run, both, neither
        )
        point = (only_second_run - only_first_run) / len(questions)
        label = f"paired Bayes, run {number + 1} only ({len(questions)} pairs)"
        lines.append(f"{label:<40}{100 * point:>7.1f}{_points((low, high)):>20}")
    lines += [
        "",
        f"{'who refused more':<30}{'shipped only':>13}{'permissive only':>17}{'p':>7}",
    ]
    only_first = sum(x and not y for x, y in zip(flat[first], flat[second]))
    only_second = sum(y and not x for x, y in zip(flat[first], flat[second]))
    lines.append(
        f"{f'by answer ({n1} pairs)':<30}{only_first:>13}{only_second:>17}"
        f"{exact_sign_test(only_first, only_second):>7.2f}"
    )
    q_first, q_second = _who_more(rates[first], rates[second])
    lines.append(
        f"{f'by question ({len(questions)})':<30}{q_first:>13}{q_second:>17}"
        f"{exact_sign_test(q_first, q_second):>7.2f}"
    )
    return "\n".join(lines) + "\n"


def render_noise(cases, runs) -> str:
    """Single runs against single runs: the same prompt twice, then the two prompts."""
    outcomes = question_outcomes(cases, runs)
    names = list(outcomes)
    questions = sorted(outcomes[names[0]])

    def flags(name, number):
        return [outcomes[name][q][number] for q in questions]

    pairs = [
        (f"{n.split()[0]}, run 1 against run 2", flags(n, 0), flags(n, 1))
        for n in names
    ]
    pairs += [
        (
            f"shipped against permissive, run {i + 1}",
            flags(names[0], i),
            flags(names[1], i),
        )
        for i in (0, 1)
    ]
    lines = [
        f"{f'single runs, {len(questions)} questions':<38}"
        + f"{'only first':>11}{'only second':>12}{'p':>7}"
    ]
    for label, a_flags, b_flags in pairs:
        _, only_a, only_b, _ = paired_counts(a_flags, b_flags)
        lines.append(
            f"{label:<38}{only_a:>11}{only_b:>12}"
            f"{exact_sign_test(only_a, only_b):>7.2f}"
        )
    return "\n".join(lines) + "\n"


def _null_slices(pairs, studies, count, size, seed):
    """Per study, the p-values of `count` slices of `size` questions when the two
    conditions are the same. Questions are drawn from the real ones with replacement
    and which condition each outcome belongs to is swapped at random."""
    rng = random.Random(seed)
    cache: dict[tuple[int, int], float] = {}
    out = []
    for _ in range(studies):
        ps = []
        for _ in range(count):
            wins = [0, 0]
            for _ in range(size):
                a, b = pairs[rng.randrange(len(pairs))]
                if rng.random() < 0.5:
                    a, b = b, a
                if a > b:
                    wins[0] += 1
                elif b > a:
                    wins[1] += 1
            key = (wins[0], wins[1])
            if key not in cache:
                cache[key] = exact_sign_test(*key)
            ps.append(cache[key])
        out.append(ps)
    return out


def render_slices(cases, runs, studies: int = 2000, counts=(1, 5, 20)) -> str:
    outcomes = question_outcomes(cases, runs)
    names, _, rates = _rates(outcomes)
    pairs = list(zip(rates[names[0]], rates[names[1]]))
    all_p = _null_slices(pairs, studies, max(counts), len(pairs), seed=0)
    lines = [
        f"{'slices compared':<18}{'any p < 0.05':>14}{'after Holm':>12}{'after BH':>10}"
    ]
    for count in counts:
        raw = holm_hits = bh_hits = 0
        for ps in all_p:
            ps = ps[:count]
            raw += min(ps) < 0.05
            holm_hits += min(holm(ps)) < 0.05
            bh_hits += min(benjamini_hochberg(ps)) < 0.05
        lines.append(
            f"{count:<18}{_pct(raw / studies):>14}{_pct(holm_hits / studies):>12}"
            f"{_pct(bh_hits / studies):>10}"
        )
    return "\n".join(lines) + "\n"


def render_plan(
    cases,
    runs,
    gaps=(0.05, 0.10, 0.15),
    sizes=(55, 110, 220, 440, 880),
) -> str:
    outcomes = question_outcomes(cases, runs)
    names, questions, rates = _rates(outcomes)
    differ = sum(x != y for x, y in zip(rates[names[0]], rates[names[1]]))
    noise = differ / len(questions)
    lines = [
        "questions where the two prompts refuse a different number of times: "
        + f"{differ} of {len(questions)}",
        "",
        f"{'questions':<12}" + "".join(f"{f'gap {100 * g:.0f} pts':>12}" for g in gaps),
    ]
    for n in sizes:
        cells = "".join(f"{_pct(power_sign_test(n, g, noise + g)):>12}" for g in gaps)
        lines.append(f"{n:<12}{cells}")
    needed = []
    for g in gaps:
        found = questions_needed(g, noise + g)
        needed.append(f"{found:>12}" if found else f"{'over 3000':>12}")
    lines.append(f"{'for 80%':<12}" + "".join(needed))
    return "\n".join(lines) + "\n"


def render_coverage(sizes=(55, 110), rates=(0.03, 0.10, 0.25, 0.50)) -> str:
    head = f"{'true rate':<11}" + "".join(
        f"{f'Wald n={n}':>13}{f'Wilson n={n}':>14}" for n in sizes
    )
    lines = [head]
    for p in rates:
        cells = ""
        for n in sizes:
            cells += f"{_pct(coverage(wald_interval, n, p)):>13}"
            cells += f"{_pct(coverage(wilson_interval, n, p)):>14}"
        lines.append(f"{_pct(p):<11}{cells}")
    return "\n".join(lines) + "\n"


ATTACK_ITEMS = "datasets/judge_items_ch16_attacks_v1.jsonl"
ATTACK_RUNS = ("runs/judge-ch16-attacks", "runs/judge-v3-attacks")


def render_attack(items, rows_a, rows_b, kind: str = "fake_source") -> str:
    """Two judge prompts against one attack: how many faulty verdicts got past."""
    faulty = {
        i["item_id"]: i["question"]
        for i in items
        if i["perturbation"] == kind and i["group"] == "planted"
    }

    def past(rows):
        return {
            (r["item_id"], r["pass"]): r["verdict"] == "supported"
            for r in rows
            if r["item_id"] in faulty and r["verdict"] is not None
        }

    a, b = past(rows_a), past(rows_b)
    keys = sorted(set(a) & set(b))
    ka, kb = sum(a[k] for k in keys), sum(b[k] for k in keys)
    only_a = sum(a[k] and not b[k] for k in keys)
    only_b = sum(b[k] and not a[k] for k in keys)
    by_question: dict[str, list[float]] = {}
    for key in keys:
        by_question.setdefault(faulty[key[0]], []).append(a[key] - b[key])
    diffs = [mean(v) for v in by_question.values()]
    q_a = sum(d > 0 for d in diffs)
    q_b = sum(d < 0 for d in diffs)
    per_q_b: dict[str, int] = {}
    for key in keys:
        per_q_b[faulty[key[0]]] = per_q_b.get(faulty[key[0]], 0) + b[key]
    q_b_any = sum(v > 0 for v in per_q_b.values())
    lines = [
        f"faulty verdicts that got past the judge: {kind}",
        f"{'':<12}{'past':>6}{'of':>5}{'Wilson':>10}",
        f"{'prompt v2':<12}{ka:>6}{len(keys):>5}{_span(wilson_interval(ka, len(keys))):>10}",
        f"{'prompt v3':<12}{kb:>6}{len(keys):>5}{_span(wilson_interval(kb, len(keys))):>10}",
        "",
        (
            f"answers: only v2 missed {only_a}, only v3 missed {only_b},"
            f" p = {exact_sign_test(only_a, only_b):.1e}"
        ),
        (
            f"questions ({len(diffs)}): v2 missed more on {q_a}, v3 on {q_b},"
            f" tied {len(diffs) - q_a - q_b}"
        ),
        f"question-level p = {exact_sign_test(q_a, q_b):.1e}",
        (
            f"questions where v3 let anything past: {q_b_any} of {len(diffs)}"
            f" ({_span(wilson_interval(q_b_any, len(diffs)))})"
        ),
        (
            f"v2 minus v3, redrawing {len(diffs)} questions: "
            f"{100 * mean(diffs):.1f} points "
            f"({_points(cluster_interval(mean, diffs))})"
        ),
    ]
    return "\n".join(lines) + "\n"


def attack_report(root=".") -> str:
    import json
    from pathlib import Path

    text = (Path(root) / ATTACK_ITEMS).read_text(encoding="utf8")
    items = [json.loads(x) for x in text.splitlines() if x.strip()]
    return render_attack(
        items,
        load_rows([Path(root) / ATTACK_RUNS[0]]),
        load_rows([Path(root) / ATTACK_RUNS[1]]),
    )


# ------------------------------------------- how good are the paired intervals?


def bayes_paired_interval(
    only_a: int,
    only_b: int,
    both: int,
    neither: int,
    draws: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """95% credible interval for (rate of a) - (rate of b) on the same questions.

    The four kinds of question, both / only a / only b / neither, get a flat
    Dirichlet prior (one imagined question of each kind). The difference is the
    share of questions only a gets minus the share only b gets."""
    rng = random.Random(seed)
    shape = (both + 1, only_a + 1, only_b + 1, neither + 1)
    diffs = []
    for _ in range(draws):
        g = [rng.gammavariate(k, 1.0) for k in shape]
        total = sum(g)
        diffs.append((g[1] - g[2]) / total)
    diffs.sort()
    return diffs[int(0.025 * draws)], diffs[int(0.975 * draws) - 1]


def paired_interval_coverage(
    n: int,
    gap: float,
    noise: float,
    both: float,
    studies: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[float, float]]:
    """Simulate `studies` experiments of n questions where the true difference in
    rates is `gap`, and report, for three ways of making the 95% interval, how often
    it contains the truth and its average width (as shares of questions).

    A question is `both`, `only a`, `only b` or `neither`. Noise is the share that
    differ from random flips; the gap comes on top, all in a's favour."""
    only_a = noise / 2 + gap
    only_b = noise / 2
    cells = [both, only_a, only_b, 1 - both - only_a - only_b]
    if min(cells) < 0:
        raise ValueError("the cells do not fit")
    rng = random.Random(seed)
    hits = {"unpaired": 0, "paired bootstrap": 0, "paired Bayes": 0}
    widths = {k: 0.0 for k in hits}
    for _ in range(studies):
        kinds = rng.choices(range(4), weights=cells, k=n)
        c = [kinds.count(i) for i in range(4)]
        a_total, b_total = c[0] + c[1], c[0] + c[2]
        made = {
            "unpaired": newcombe_difference(a_total, n, b_total, n),
            "paired bootstrap": cluster_interval(
                mean,
                [1.0] * c[1] + [-1.0] * c[2] + [0.0] * (c[0] + c[3]),
                resamples=600,
                seed=rng.randrange(10**6),
            ),
            "paired Bayes": bayes_paired_interval(
                c[1], c[2], c[0], c[3], draws=1500, seed=rng.randrange(10**6)
            ),
        }
        for name, (low, high) in made.items():
            hits[name] += low <= gap <= high
            widths[name] += high - low
    return {k: (hits[k] / studies, widths[k] / studies) for k in hits}


def render_paired_coverage(n: int = 55, noise: float = 0.09, both: float = 0.2) -> str:
    lines = [f"{'':<16}{'true gap':>8}" + "".join(f"{m:>18}" for m in _METHODS)]
    for gap in (0.0, 0.10):
        got = paired_interval_coverage(n, gap, noise, both)
        cells = "".join(
            f"{f'{_pct(got[m][0])} of {100 * got[m][1]:.0f} pts':>18}" for m in _METHODS
        )
        lines.append(f"{f'{n} questions':<16}{_pct(gap):>8}{cells}")
    return "\n".join(lines) + "\n"


_METHODS = ("unpaired", "paired bootstrap", "paired Bayes")
