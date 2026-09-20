import pytest

from agent_evals.stats import min_cases_for_perfect_run, percentile, wilson_interval


def _close(actual: tuple[float, float], low: float, high: float, tol: float = 5e-4):
    assert actual[0] == pytest.approx(low, abs=tol)
    assert actual[1] == pytest.approx(high, abs=tol)


def test_six_of_six_is_far_from_certain():
    _close(wilson_interval(6, 6), 0.610, 1.0)


@pytest.mark.parametrize(
    "successes,n,low,high",
    [
        (20, 20, 0.839, 1.0),
        (50, 50, 0.929, 1.0),
        (100, 100, 0.963, 1.0),
        (92, 100, 0.850, 0.959),
        (917, 1000, 0.898, 0.9325),
    ],
)
def test_known_intervals(successes, n, low, high):
    _close(wilson_interval(successes, n), low, high)


def test_zero_violations_in_a_thousand_still_leaves_an_upper_bound():
    low, high = wilson_interval(0, 1000)
    assert low == pytest.approx(0.0, abs=1e-9)
    assert high == pytest.approx(0.0038, abs=2e-4)


def test_a_lower_bound_gate_at_eighty_percent_needs_sixteen_cases():
    assert min_cases_for_perfect_run(0.8) == 16
    assert wilson_interval(15, 15)[0] < 0.8 <= wilson_interval(16, 16)[0]


def test_invalid_inputs_are_rejected():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(7, 6)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4], 0.5) == 2.5
    assert percentile([5.0], 0.95) == 5.0
