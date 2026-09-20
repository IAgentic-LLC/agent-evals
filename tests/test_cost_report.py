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
    assert rows["3.6-flash"] == "3.6-flash 126 84 1.8k/0.7k 0.0040 0.0060 7.0 14.6"
    assert rows["2.5-flash"] == "2.5-flash 126 95 0.8k/0.4k 0.0012 0.0016 2.3 6.2"
    assert rows["3.6 + stall guard"].split()[5] == "111"


def test_the_first_attempt_counted_only_the_tokens_the_model_wrote():
    cases, prices, _ = _recorded()
    first = [
        (label, model, read_traces(ROOT / "runs" / run / "traces.jsonl"))
        for label, model, run in cli.COST_RUNS_FIRST
    ]
    rows = {
        line.split("  ")[0].strip(): _flat(line)
        for line in cost.render_models(cases, first, prices).splitlines()[1:]
    }
    assert rows["3.6-flash"] == "3.6-flash 126 84 1.7k/0.1k 0.0019 0.0028 9.0 21.1"
    assert rows["2.5-flash"] == "2.5-flash 126 98 0.8k/0.1k 0.0004 0.0006 2.5 5.9"


def test_thinking_is_most_of_what_two_of_the_models_bill_and_none_for_the_third():
    _, prices, conditions = _recorded()
    rows = {
        line.split("  ")[0].strip(): _flat(line)
        for line in cost.render_thinking(conditions, prices).splitlines()[1:]
    }
    assert rows["3.6-flash"] == "3.6-flash 145 567 80% 0.0019 0.0040"
    assert rows["2.5-flash"] == "2.5-flash 82 291 78% 0.0004 0.0012"
    assert rows["3.5-flash-lite"] == "3.5-flash-lite 131 0 0% 0.0007 0.0007"


def test_the_written_only_cost_of_the_metered_run_matches_the_first_attempts_meter():
    _, prices, conditions = _recorded()
    lines = cost.render_thinking(conditions, prices).splitlines()[1:]
    written = {line.split("  ")[0].strip(): float(line.split()[-2]) for line in lines}
    assert written["3.6-flash"] == 0.0019
    assert written["2.5-flash"] == 0.0004


def test_the_frontier_drops_only_the_default_model():
    cases, prices, conditions = _recorded()
    lines = cost.render_frontier(cases, conditions, prices).splitlines()[1:]
    on = {line.split("  ")[0].strip() for line in lines if line.endswith("yes")}
    assert on == {"3.5-flash-lite", "2.5-flash", "3.6 + stall guard"}


def test_three_in_ten_dollars_of_the_default_models_spend_went_to_runs_that_looped():
    cases, prices, conditions = _recorded()
    _, model, traces = conditions[0]
    table = _flat(cost.render_spend(cases, traces, prices[model]))
    assert "met 84 0.0040 66%" in table
    assert "tool loop 29 0.0035 20%" in table
    assert "handoff loop 7 0.0069 10%" in table


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
        "3.6 + stall guard 14 2 26 0.004 +21.4 (+10.3 to +33.3)"
    )
    assert rows["2.5-flash"] == "2.5-flash 12 7 23 0.359 +8.7 (-6.3 to +23.0)"
    assert rows["3.5-flash-lite"].startswith("3.5-flash-lite 8 6 28 0.791")


def test_the_same_condition_run_twice_is_never_told_apart_from_itself():
    cases, _, conditions = _recorded()
    pairs = [
        (label, read_traces(ROOT / "runs" / first / "traces.jsonl"), traces)
        for (label, _, first), (_, _, traces) in zip(
            cli.COST_RUNS_FIRST, conditions, strict=True
        )
    ]
    text = cost.render_repeat(cases, pairs)
    rows = {line.split("  ")[0].strip(): _flat(line) for line in text.splitlines()[1:]}
    assert rows["3.6-flash"] == "3.6-flash 1 3 38 0.625 +0.0 (-4.8 to +6.3)"
    assert rows["2.5-flash"] == "2.5-flash 4 7 31 0.549 -2.4 (-8.7 to +4.0)"
    for line in text.splitlines()[1:]:
        assert float(line.split()[-6]) > 0.05


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

    parts = ("models", "frontier", "spend", "latency", "plan", "paired")
    for part in (*parts, "thinking", "repeat", "bill", "invariants"):
        argv = ["agent-evals", "cost", "--part", part]
        old, sys.argv = sys.argv, argv
        try:
            assert cli.main() == 0
        finally:
            sys.argv = old
        out = capsys.readouterr().out
        assert out and all(len(line) <= 78 for line in out.splitlines())


def test_the_bill_for_the_four_metered_conditions_is_the_sum_of_their_runs():
    _, prices, conditions = _recorded()
    rows = {
        line.split("  ")[0].strip(): _flat(line)
        for line in cost.render_bill(conditions, prices).splitlines()[1:]
    }
    assert rows["3.6-flash"] == "3.6-flash 126 0.50"
    assert rows["all"] == "all 504 1.30"


def test_the_first_attempt_bill_is_a_floor_because_it_left_thinking_out():
    _, prices, _ = _recorded()
    first = [
        (label, model, read_traces(ROOT / "runs" / run / "traces.jsonl"))
        for label, model, run in cli.COST_RUNS_FIRST
    ]
    assert _flat(cost.render_bill(first, prices).splitlines()[-1]) == "all 504 0.63"


def test_the_latency_table_shows_a_tail_the_mean_does_not():
    cases, _, conditions = _recorded()
    label, _, traces = conditions[2]
    assert label == "2.5-flash"
    row = _flat(cost.render_latency(cases, traces).splitlines()[1])
    assert row.startswith("all 126 3.1 2.3 6.2 20.0 ")


def test_the_invariant_report_finds_the_one_run_that_changed_the_customer_spelling():
    cases, _, conditions = _recorded()
    lines = cost.render_invariants(cases, conditions).splitlines()
    assert _flat(lines[1]) == "3.6-flash 126 0"
    assert _flat(lines[2]) == "3.5-flash-lite 126 1"
    assert _flat(lines[3]) == "2.5-flash 126 0"
    assert _flat(lines[4]) == "3.6 + stall guard 126 0"
    assert lines[5] == (
        "  3.5-flash-lite, HO-008 run 1: look_up_invoice cust_311, ticket has cust-311"
    )
    assert len(lines) == 6
