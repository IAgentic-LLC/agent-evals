"""Compare my small statistics with scipy and statsmodels (chapter 18).

Usage: uv run --with scipy --with statsmodels \
    python scripts/check_compare_against_libraries.py
"""

import random

from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.proportion import confint_proportions_2indep

from agent_evals import compare


def main() -> None:
    rng = random.Random(3)
    worst = {"exact test": 0.0, "Holm": 0.0, "Benjamini-Hochberg": 0.0}
    worst["Newcombe interval"] = 0.0
    for _ in range(300):
        a, b = rng.randint(0, 30), rng.randint(0, 30)
        if a + b:
            theirs = binomtest(min(a, b), a + b, 0.5).pvalue
            worst["exact test"] = max(
                worst["exact test"], abs(compare.exact_sign_test(a, b) - theirs)
            )
        ps = [rng.random() ** 2 for _ in range(rng.randint(2, 20))]
        for name, mine, method in (
            ("Holm", compare.holm, "holm"),
            ("Benjamini-Hochberg", compare.benjamini_hochberg, "fdr_bh"),
        ):
            theirs = multipletests(ps, method=method)[1]
            gap = max(abs(x - y) for x, y in zip(mine(ps), theirs))
            worst[name] = max(worst[name], gap)
        n1, n2 = rng.randint(20, 200), rng.randint(20, 200)
        k1, k2 = rng.randint(1, n1 - 1), rng.randint(1, n2 - 1)
        low, high = confint_proportions_2indep(k1, n1, k2, n2, method="newcomb")
        mine = compare.newcombe_difference(k1, n1, k2, n2)
        worst["Newcombe interval"] = max(
            worst["Newcombe interval"], abs(mine[0] - low), abs(mine[1] - high)
        )
    for name, gap in worst.items():
        print(f"{name:<20} largest difference {gap:.1e} over 300 random cases")


if __name__ == "__main__":
    main()
