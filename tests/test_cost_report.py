"""Chapter 22: cost, latency and the price of an evaluation, from the recorded runs.

The dollar figures are checked by hand arithmetic on a synthetic trace, and the tables
are pinned against the four recorded runs, so a change to a price, an outcome rule or a
resampling seed shows up here first.
"""

from pathlib import Path

import pytest

from agent_evals import cli, cost
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]


def _flat(text):
    return " ".join(text.split())


def _trace(case_id="T", trial=1, error=None, latency=1.0, tokens=(0, 0)):
    return Trace(
        case_id=case_id,
        trial=trial,
        adapter="a",
        error=error,
        latency_s=latency,
        usage={"model_calls": 1, "input_tokens": tokens[0], "output_tokens": tokens[1]},
    )


def _recorded():
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    prices = cost.load_prices(ROOT / "config/prices.yaml")
    conditions = [
        (label, model, read_traces(ROOT / "runs" / run / "traces.jsonl"))
        for label, model, run in cli.COST_RUNS
    ]
    return cases, prices, conditions


def test_the_price_file_names_its_source_and_the_day_it_was_checked():
    text = (ROOT / "config/prices.yaml").read_text(encoding="utf8")
    assert "source: https://ai.google.dev/gemini-api/docs/pricing" in text
    assert "checked: 2026-09-21" in text


def test_prices_load_as_input_and_output_dollars_per_million_tokens():
    prices = cost.load_prices(ROOT / "config/prices.yaml")
    assert prices["gemini-3.6-flash"] == (0.75, 3.75)
    assert prices["gemini-2.5-flash"] == (0.30, 2.50)


def test_a_run_costs_its_input_tokens_at_one_price_and_its_output_tokens_at_another():
    trace = _trace(tokens=(2_000_000, 1_000_000))
    assert cost.run_cost(trace, (0.75, 3.75)) == pytest.approx(2 * 0.75 + 3.75)
    assert cost.run_cost(_trace(tokens=(1000, 100)), (1.0, 2.0)) == pytest.approx(
        0.0012
    )


def test_a_run_recorded_without_tokens_has_no_cost_and_asking_is_an_error():
    trace = Trace(case_id="T", adapter="a")
    with pytest.raises(ValueError, match="no recorded tokens"):
        cost.run_cost(trace, (1.0, 1.0))


def test_the_frontier_keeps_only_points_that_nothing_beats_on_both_axes():
    points = [
        ("cheap", 1.0, 0.6),
        ("good", 3.0, 0.9),
        ("dear and poor", 3.0, 0.6),
        ("same as cheap", 1.0, 0.6),
    ]
    assert cost.pareto(points) == ["cheap", "good", "same as cheap"]


def test_a_latency_interval_redraws_whole_tickets_and_covers_the_percentile():
    traces = [_trace(f"T{i}", t, latency=float(i)) for i in range(20) for t in (1, 2)]
    low, high = cost.latency_interval(traces, 0.5)
    assert low <= 9.5 <= high
    assert cost.latency_interval(traces, 0.5) == (low, high)


def test_the_models_table_is_pinned_to_the_four_recorded_runs():
    cases, prices, conditions = _recorded()
    rows = {
        line.split("  ")[0].strip(): _flat(line)
        for line in cost.render_models(cases, conditions, prices).splitlines()[1:]
    }
    assert rows["3.6-flash"] == "3.6-flash 126 84 1.7k/0.1k 0.0019 0.0028 9.0 21.1"
    assert rows["2.5-flash"] == "2.5-flash 126 98 0.8k/0.1k 0.0004 0.0006 2.5 5.9"
    assert rows["3.6 + stall guard"].split()[5] == "114"


def test_the_frontier_holds_the_cheapest_and_the_guarded_run_and_no_other():
    cases, prices, conditions = _recorded()
    lines = cost.render_frontier(cases, conditions, prices).splitlines()[1:]
    on = {line.split("  ")[0].strip() for line in lines if line.endswith("yes")}
    assert on == {"2.5-flash", "3.6 + stall guard"}


def test_more_than_a_quarter_of_the_default_models_spend_went_to_tool_loops():
    cases, prices, conditions = _recorded()
    _, model, traces = conditions[0]
    table = _flat(cost.render_spend(cases, traces, prices[model]))
    assert "met 84 0.0017 61%" in table
    assert "tool loop 27 0.0024 27%" in table


def test_the_spend_shares_add_to_one_hundred_percent_within_rounding():
    cases, prices, conditions = _recorded()
    _, model, traces = conditions[0]
    lines = cost.render_spend(cases, traces, prices[model]).splitlines()[1:]
    shares = [int(line.split()[-1].rstrip("%")) for line in lines]
    assert abs(sum(shares) - 100) <= 2


def test_the_paired_table_pins_the_counts_and_the_intervals():
    cases, _, conditions = _recorded()
    base = conditions[0][2]
    others = [(c[0], c[2]) for c in conditions[1:]]
    rows = {
        line.split("  ")[0].strip(): _flat(line)
        for line in cost.render_paired(cases, base, others, "3.6-flash").splitlines()[
            1:
        ]
    }
    assert rows["3.6 + stall guard"] == (
        "3.6 + stall guard 13 0 29 <0.001 +23.8 (+12.7 to +35.7)"
    )
    assert rows["2.5-flash"] == "2.5-flash 13 6 23 0.167 +11.1 (-4.0 to +26.2)"
    assert rows["3.5-flash-lite"].startswith("3.5-flash-lite 8 6 28 0.791")


def test_a_cheaper_model_that_is_not_shown_worse_is_not_shown_better_either():
    cases, _, conditions = _recorded()
    text = cost.render_paired(
        cases, conditions[0][2], [(conditions[2][0], conditions[2][2])], "3.6-flash"
    )
    low = float(text.split("(")[1].split(" to ")[0])
    assert low < 0


def test_the_plan_prices_two_runs_of_every_ticket_the_power_plan_asked_for():
    _, prices, conditions = _recorded()
    text = cost.render_plan(conditions[:1], prices)
    per_run = sum(
        cost.run_cost(t, prices["gemini-3.6-flash"]) for t in conditions[0][2]
    )
    per_run /= len(conditions[0][2])
    assert f"{2 * 161 * per_run:.2f}" in text
    assert "88" in text and "161" in text and "466" in text


def test_every_cost_part_runs_from_the_command_line(capsys):
    import sys

    for part in ("models", "frontier", "spend", "latency", "plan", "paired"):
        argv = ["agent-evals", "cost", "--part", part]
        old, sys.argv = sys.argv, argv
        try:
            assert cli.main() == 0
        finally:
            sys.argv = old
        out = capsys.readouterr().out
        assert out and all(len(line) <= 78 for line in out.splitlines())
