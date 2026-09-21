"""Chapter 24: watching a deployed product from recorded runs.

The monitors are checked on small hand-made streams where the answer is known. The tables
are pinned against the recorded runs with seeded draws, so they are the same every time.
"""

import contextlib
import io
import random
from functools import cache
from pathlib import Path

from agent_evals import cli, online

ROOT = Path(__file__).resolve().parents[1]


def _flat(text):
    return " ".join(text.split())


@cache
def _render(part):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["online", "--part", part]) == 0
    text = buf.getvalue()
    assert all(len(line) <= 78 for line in text.splitlines())
    return text


def _days(rates, n=40):
    return [(round(r * n), n) for r in rates]


# ------------------------------------------------------------------ the monitors


def test_no_monitor_alarms_on_a_stream_that_never_moves():
    base = _days([0.8] * 14)
    for monitor in online.MONITORS:
        assert online.first_alarm(monitor, base, _days([0.8] * 30)) is None


def test_every_monitor_alarms_on_the_first_day_of_a_large_fall():
    base = _days([0.9] * 14)
    for monitor in online.MONITORS:
        assert online.first_alarm(monitor, base, _days([0.5] * 10)) == 1


def test_the_threshold_wants_a_fall_of_ten_points_and_the_chart_wants_three_sigmas():
    base = _days([0.8] * 14, n=400)
    small = _days([0.72] * 10, n=400)
    assert online.first_alarm("threshold", base, small) is None
    assert online.first_alarm("shewhart", base, small) == 1


def test_cusum_catches_a_small_shift_the_chart_misses_by_adding_days_up():
    base = _days([0.8] * 14, n=100)
    slow = _days([0.74] * 30, n=100)
    assert online.first_alarm("shewhart", base, slow) is None
    day = online.first_alarm("cusum", base, slow)
    assert day is not None and day > 1


def test_a_day_with_no_tickets_is_skipped_and_a_perfect_baseline_does_not_divide_by_zero():
    base = _days([1.0] * 14)
    watch = [(0, 0), (30, 40), (39, 40)]
    for monitor in online.MONITORS:
        assert online.first_alarm(monitor, base, watch) in (None, 2, 3)


def test_a_day_is_n_tickets_each_with_one_of_its_own_recorded_outcomes():
    pop = online.Population([("a", [True, True]), ("b", [False])])
    rows = pop.day(random.Random(1), 200)
    assert len(rows) == 200
    assert all(ok for stratum, ok in rows if stratum == "a")
    assert not any(ok for stratum, ok in rows if stratum == "b")
    assert abs(pop.rate - 2 / 3) < 1e-9


def test_a_stratum_keeps_only_that_slices_tickets():
    pop = online.Population([("a", [True]), ("b", [False])])
    k, n = online.stream([pop], random.Random(2), 100, "a")[0]
    assert k == n and 20 < n < 80


def test_the_streams_are_seeded_so_a_study_gives_the_same_answer_twice():
    pop = online.Population([("a", [True, False]), ("b", [True])])
    a = online.false_alarms(pop, 20, 30, seed=5)
    assert a == online.false_alarms(pop, 20, 30, seed=5)


def test_the_distance_between_two_mixes_runs_from_zero_to_one():
    same = {"a": 0.5, "b": 0.5}
    assert online.js_distance(same, same) == 0
    assert abs(online.js_distance({"a": 1.0}, {"b": 1.0}) - 1.0) < 1e-9


# ------------------------------------------------------------ the recorded runs


def test_the_proxy_table_says_no_error_runs_a_few_points_above_met_and_keeps_the_order():
    text = _flat(_render("proxy"))
    assert "default 126 71.4% 66.7% 4.8 6" in text
    assert "default with the stall guard 126 93.7% 88.1% 5.6 7" in text
    assert "2.5-flash 126 82.5% 75.4% 7.1 9" in text
    assert "change set as shipped 120 23.3% 23.3% 0.0 0" in text


def test_the_same_product_scores_67_and_23_on_two_sets_and_topic_other_barely_moves():
    text = _flat(_render("mix"))
    assert "other 20.8% (24) 20.4% (54)" in text
    assert "login 0.0% (12) 0.0% (30)" in text
    assert "all 66.7% (126) 23.3% (120)" in text
    assert "distance between the specialist mixes: 0.63" in text
    assert "held-out rates on the change set runbook_topic mix: 33.4%" in text


def test_the_review_table_pins_the_interval_and_the_chance_of_seeing_a_rare_problem():
    text = _flat(_render("review"))
    assert "100 21 to 39% 63% 18%" in text
    assert "400 25 to 34% 98% 55%" in text


def test_the_false_alarm_table_is_pinned_and_a_fixed_threshold_fails_at_low_traffic():
    text = _flat(_render("false-alarms"))
    assert "10 100% 11% 21%" in text
    assert "160 13% 8% 16%" in text


def test_the_falls_table_is_pinned_and_a_large_fall_is_caught_on_the_first_day():
    text = _flat(_render("falls"))
    assert "guard -> default 40 96% day 1 96% day 1 97% day 1" in text
    assert "2.5-flash -> lite 40 73% day 2 56% day 5 88% day 4" in text


def test_a_change_of_mix_alarms_the_pooled_monitor_and_not_the_one_on_topic_other():
    text = _flat(_render("mix-shift"))
    assert "all tickets 40 63% day 1 98% day 1 98% day 1" in text
    assert "topic other 40 9% day 6 0% day - 4% day 12" in text


def test_the_shadow_table_is_pinned():
    text = _flat(_render("shadow"))
    assert "84 97% 43% 100%" in text
    assert "168 100% 77% 100%" in text
