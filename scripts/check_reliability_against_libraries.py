"""Compare my pass@k, pass^k and variance components with other code (chapter 19).

pass@k and pass^k are checked against a brute-force average over every choice of k of
the n recorded trials, which is their definition. The mean squares are checked against
scipy's one-way ANOVA F statistic.

Usage: uv run --with scipy python scripts/check_reliability_against_libraries.py
"""

import random
from itertools import combinations

from scipy.stats import f_oneway

from agent_evals import reliability


def brute(outcomes: list[bool], k: int) -> tuple[float, float]:
    """(pass@k, pass^k) as the average over every subset of k trials."""
    subsets = list(combinations(outcomes, k))
    at = sum(any(s) for s in subsets) / len(subsets)
    hat = sum(all(s) for s in subsets) / len(subsets)
    return at, hat


def main() -> None:
    rng = random.Random(4)
    worst = 0.0
    for _ in range(300):
        n = rng.randint(2, 9)
        outcomes = [rng.random() < rng.random() for _ in range(n)]
        k = rng.randint(1, n)
        at, hat = brute(outcomes, k)
        c = sum(outcomes)
        worst = max(
            worst,
            abs(reliability.pass_at(c, n, k) - at),
            abs(reliability.pass_hat(c, n, k) - hat),
        )
    print(f"pass@k and pass^k: largest difference {worst:.1e} over 300 cases")
    worst = 0.0
    done = 0
    while done < 300:
        cases, trials = rng.randint(3, 30), rng.randint(2, 6)
        p = [rng.random() for _ in range(cases)]
        outs = {
            str(i): [rng.random() < p[i] for _ in range(trials)] for i in range(cases)
        }
        if all(len(set(v)) == 1 for v in outs.values()):
            continue
        msb, msw, _, _ = reliability.variance_components(outs)
        if msw == 0:
            continue
        groups = [[float(x) for x in v] for v in outs.values()]
        f_theirs = f_oneway(*groups).statistic
        worst = max(worst, abs(msb / msw - f_theirs))
        done += 1
    print(f"one-way ANOVA F statistic: largest difference {worst:.1e} over 300 cases")


if __name__ == "__main__":
    main()
