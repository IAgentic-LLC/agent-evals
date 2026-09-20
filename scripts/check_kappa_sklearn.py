"""Compare my kappa with scikit-learn's on 300 random cases (chapter 17).

Usage: uv run --with scikit-learn python scripts/check_kappa_sklearn.py
"""

import random

from sklearn.metrics import cohen_kappa_score

from agent_evals import agreement


def main() -> None:
    rng = random.Random(5)
    worst, cases = 0.0, 0
    while cases < 300:
        n = rng.randint(10, 80)
        classes = rng.choice([2, 3])
        a = [rng.randrange(classes) for _ in range(n)]
        b = [x if rng.random() < 0.7 else rng.randrange(classes) for x in a]
        if len(set(a)) < 2 and len(set(b)) < 2:
            continue
        cases += 1
        mine = agreement.cohen_kappa(a, b)
        theirs = cohen_kappa_score(a, b)
        worst = max(worst, abs(mine - theirs))
    print(f"{cases} cases, largest difference {worst:.1e}")


if __name__ == "__main__":
    main()
