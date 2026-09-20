"""How much two raters agree, and what an imperfect judge does to a measured rate (chapter 17).

A rater is anything that gives each item a label: a person, a judge, a second pass of
either. Agreement beyond chance is Cohen's kappa. A judge with known sensitivity and
specificity, measured against people on a labeled calibration set, can have its measured
rate corrected, and the correction comes with an interval that includes the calibration
set's own uncertainty.
"""

import hashlib
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any


def percent_agreement(a: list[str], b: list[str]) -> float:
    return sum(x == y for x, y in zip(a, b)) / len(a)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Agreement beyond what chance would give, from each rater's own label rates.
    1 is perfect, 0 is what independent raters with the same rates would get."""
    n = len(a)
    observed = percent_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / (n * n)
    if expected == 1:
        return 1.0 if observed == 1 else 0.0
    return (observed - expected) / (1 - expected)


def pabak(a: list[str], b: list[str]) -> float:
    """Kappa if the two labels were equally common: 2 times observed agreement, minus 1.
    It does not shrink when one label is rare, which kappa does."""
    return 2 * percent_agreement(a, b) - 1


def bootstrap_interval(
    stat,
    *columns: list[Any],
    groups: list[str] | None = None,
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, float]:
    """A 95% interval for a statistic of parallel columns, resampling whole groups.

    Items that share a group, such as answers to the same question, are drawn together,
    because they are not independent evidence. With no groups every item is its own.
    """
    rng = random.Random(seed)
    n = len(columns[0])
    keys = groups or [str(i) for i in range(n)]
    members: dict[str, list[int]] = {}
    for i, key in enumerate(keys):
        members.setdefault(key, []).append(i)
    names = list(members)
    values = []
    for _ in range(resamples):
        picked = [i for name in rng.choices(names, k=len(names)) for i in members[name]]
        try:
            values.append(stat(*[[c[i] for i in picked] for c in columns]))
        except ZeroDivisionError:
            continue
    values.sort()
    if not values:
        return (float("nan"), float("nan"))
    return values[int(0.025 * len(values))], values[int(0.975 * len(values)) - 1]


def sensitivity_specificity(
    truth: list[bool], flagged: list[bool]
) -> tuple[float, float]:
    """Of the truly bad items, the share the judge flagged. Of the good, the share it left."""
    bad = [f for t, f in zip(truth, flagged) if t]
    good = [f for t, f in zip(truth, flagged) if not t]
    return sum(bad) / len(bad), sum(not f for f in good) / len(good)


def rogan_gladen(apparent: float, sens: float, spec: float) -> float:
    """The true rate that would give the rate a judge with this sensitivity and
    specificity reports. Clipped to 0..1, because sampling noise can push it outside."""
    denominator = sens + spec - 1
    if denominator <= 0:
        raise ZeroDivisionError("a judge no better than a coin has nothing to correct")
    return min(1.0, max(0.0, (apparent + spec - 1) / denominator))


def calibrated_rate(
    cal_truth: list[bool],
    cal_flagged: list[bool],
    cal_groups: list[str],
    test_flagged: list[bool],
    test_groups: list[str],
    resamples: int = 4000,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """The corrected rate of bad items among the test items, with an interval that
    redraws both the calibration set and the test set."""
    sens, spec = sensitivity_specificity(cal_truth, cal_flagged)
    point = rogan_gladen(sum(test_flagged) / len(test_flagged), sens, spec)
    rng = random.Random(seed)

    def clusters(groups):
        out: dict[str, list[int]] = {}
        for i, g in enumerate(groups):
            out.setdefault(g, []).append(i)
        return out

    cal, test = clusters(cal_groups), clusters(test_groups)
    values = []
    for _ in range(resamples):
        c = [i for g in rng.choices(list(cal), k=len(cal)) for i in cal[g]]
        t = [i for g in rng.choices(list(test), k=len(test)) for i in test[g]]
        truth = [cal_truth[i] for i in c]
        if not any(truth) or all(truth):
            continue
        try:
            s, p = sensitivity_specificity(truth, [cal_flagged[i] for i in c])
            values.append(rogan_gladen(sum(test_flagged[i] for i in t) / len(t), s, p))
        except ZeroDivisionError:
            continue
    values.sort()
    if not values:
        return point, (float("nan"), float("nan"))
    return point, (
        values[int(0.025 * len(values))],
        values[int(0.975 * len(values)) - 1],
    )


def group_of(text: str, buckets: int = 2) -> int:
    """A stable bucket for a question, so all answers to it land in the same half."""
    return int(hashlib.sha256(text.encode("utf8")).hexdigest(), 16) % buckets


def planning_widths(
    sens: float,
    spec: float,
    prevalence: float,
    test_size: int,
    sizes: tuple[int, ...],
    resamples: int = 400,
    seed: int = 0,
) -> list[tuple[int, float]]:
    """How wide the interval on the corrected rate is, for calibration sets of each size,
    when a judge with this sensitivity and specificity is used on `test_size` items.

    A simulation: draw a calibration set of the given size with the assumed prevalence,
    draw judge flags, and repeat, taking the middle 95% of the corrected rates."""
    rng = random.Random(seed)
    out = []
    for size in sizes:
        rates = []
        for _ in range(resamples):
            truth = [rng.random() < prevalence for _ in range(size)]
            flags = [rng.random() < (sens if t else 1 - spec) for t in truth]
            test_truth = [rng.random() < prevalence for _ in range(test_size)]
            test_flags = [rng.random() < (sens if t else 1 - spec) for t in test_truth]
            if not any(truth) or all(truth):
                continue
            s, p = sensitivity_specificity(truth, flags)
            try:
                rates.append(rogan_gladen(sum(test_flags) / test_size, s, p))
            except ZeroDivisionError:
                continue
        rates.sort()
        low, high = rates[int(0.025 * len(rates))], rates[int(0.975 * len(rates)) - 1]
        out.append((size, high - low))
    return out


def load_labels(path: str | Path) -> dict[str, str]:
    """A label file: one JSON object per line with `item_id` and `label`."""
    lines = Path(path).read_text(encoding="utf8").splitlines()
    rows = [json.loads(x) for x in lines if x.strip()]
    return {r["item_id"]: r["label"] for r in rows}
