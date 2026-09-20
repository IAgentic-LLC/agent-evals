"""Small statistics helpers. Nothing here needs a dependency beyond the standard library."""

from math import sqrt
from statistics import NormalDist


def wilson_interval(
    successes: int, n: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Preferred over the plain normal (Wald) interval at small n and at rates near
    0 or 1, which is exactly where agent evaluations live.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def min_cases_for_perfect_run(target_lower: float, confidence: float = 0.95) -> int:
    """Smallest n for which n out of n passing puts the lower bound at or above target.

    A gate of the form "the lower bound must be at least 80%" cannot be passed by
    any run with fewer cases than this, however well the agent behaves.
    """
    for n in range(1, 100_000):
        if wilson_interval(n, n, confidence)[0] >= target_lower:
            return n
    raise ValueError("target is not reachable")


def percentile(values: list[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 1]."""
    if not values:
        raise ValueError("no values")
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lower = int(k)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (k - lower)
