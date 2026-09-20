"""Chapter 20: routing and handoffs in the triage product, from the recorded runs."""

from pathlib import Path

from agent_evals import cli, reliability, routing
from agent_evals.runner import load_cases, read_traces

ROOT = Path(__file__).resolve().parents[1]


def _flat(text):
    return " ".join(text.split())


def _held():
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_heldout_v1.jsonl")
    }
    every = [
        t
        for n in reliability.TRIAL_RUNS
        for t in read_traces(ROOT / "runs" / n / "traces.jsonl")
    ]
    five = read_traces(ROOT / "runs/triage-reliability-5x/traces.jsonl")
    return cases, every, five


def _routing_set():
    cases = {
        c.case_id: c for c in load_cases(ROOT / "datasets/triage_routing_v1.jsonl")
    }
    return cases, read_traces(ROOT / "runs/triage-routing-3x/traces.jsonl")


def test_no_run_ever_finished_on_a_wrong_specialist():
    cases, every, _ = _held()
    text = _flat(routing.render_specialists(cases, every))
    assert "billing 13 104 102 0 2 12 of 13" in text
    assert "security 12 96 96 0 0 12 of 12" in text
    assert "technical 17 136 38 0 98 2 of 17" in text


def test_every_clear_misfile_was_handed_on_and_to_the_right_place():
    cases, _, five = _held()
    text = _flat(routing.render_handoffs(cases, five))
    assert "category wrong 6 30 30 30 3" in text
    assert "category right 36 180 7 - 7" in text


def test_the_handoff_loops_in_the_five_new_trials_are_on_four_tickets():
    cases, _, five = _held()
    text = _flat(routing.render_loops(cases, five))
    for line in ("HO-014", "HO-015", "HO-036", "HO-039 technical technical 5"):
        assert line in text


def test_the_routing_set_has_24_tickets_of_three_kinds():
    cases, runs = _routing_set()
    assert len(cases) == 24 and len(runs) == 72
    kinds = [c.slices["kind"] for c in cases.values()]
    assert (kinds.count("misfiled"), kinds.count("boundary")) == (12, 8)
    assert kinds.count("distractor") == 4


def test_the_routing_set_by_kind_of_ticket():
    cases, runs = _routing_set()
    text = _flat(routing.render_kinds(cases, runs))
    assert "misfiled 12 36 36 26 26 0 10 0" in text
    assert "boundary 8 24 17 7 7 0 17 14" in text
    assert "distractor 4 12 0 8 8 0 4 0" in text


def test_the_routing_set_by_ticket_counts_looping_tickets():
    cases, runs = _routing_set()
    text = _flat(routing.render_kind_tickets(cases, runs))
    assert "misfiled 12 8 of 12 39-86% 0 of 12 0-24% 12 of 12 76-100%" in text
    assert "boundary 8 2 of 8 7-59% 5 of 8 31-86% 6 of 8 41-93%" in text


def test_the_boundary_tickets_that_loop_are_the_ones_two_specialists_disown():
    cases, runs = _routing_set()
    text = _flat(routing.render_tickets(cases, runs, "boundary"))
    assert "RT-016 technical technical loop, loop, loop" in text
    assert "RT-017 security billing loop, loop, loop" in text
    assert "RT-014 security security security, security, security" in text


def test_the_two_specialists_each_say_it_belongs_to_the_other():
    cases, runs = _routing_set()
    text = _flat(routing.render_conversation(cases, runs, "RT-016"))
    assert "1. technical asks for billing:" in text
    assert "2. billing asks for technical:" in text


def test_the_reports_fit_the_page(capsys):
    parts = ("specialists", "handoffs", "loops", "kinds", "by-ticket")
    for part in (*parts, "misfiled", "boundary"):
        assert cli.main(["routing", "--part", part]) == 0
    assert (
        cli.main(["routing", "--part", "conversation", "--tickets", "RT-016", "RT-017"])
        == 0
    )
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line


def test_every_tool_loop_in_the_routing_set_is_on_a_ticket_that_belongs_to_technical():
    cases, runs = _routing_set()
    text = _flat(routing.render_owners(cases, runs))
    assert "billing 21 16 0 5" in text and "security 27 21 0 6" in text
    assert "technical 24 4 17 3" in text
