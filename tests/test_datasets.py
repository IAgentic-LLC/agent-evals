"""Chapter 4: the held-out test set, its checks, and what the first run showed.

No API key needed: the datasets and the recorded runs.
"""

import hashlib
import json
from collections import Counter
from pathlib import Path

from agent_evals.cli import main
from agent_evals.dataset import (
    Neighbor,
    check_cases,
    jaccard,
    near_dev_flags,
    nearest_lexical,
    repeated_subjects,
    slice_counts,
    within_set_pairs,
)
from agent_evals.runner import load_cases, read_traces
from agent_evals.schema import EvalCase
from agent_evals.scorecard import build_scorecard, render_by_slice

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "datasets"
V1 = DATA / "triage_heldout_v1.jsonl"
V2 = DATA / "triage_heldout_v2.jsonl"
SIX = DATA / "triage_book3_six.jsonl"
RUN = ROOT / "runs" / "triage-heldout-v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(**changes) -> EvalCase:
    base = {
        "case_id": "X-1",
        "input": {"subject": "s", "body": "b", "category": "billing"},
        "expected": {"handled_by": "billing", "required_actions": ["look_up_invoice"]},
        "slices": {"adversarial": "no"},
        "split": "test",
        "canary": "canary",
    }
    base.update(changes)
    return EvalCase(**base)


def test_the_published_versions_are_frozen():
    assert _sha(V1).startswith("27c664be86344c9f")
    assert _sha(V2).startswith("6cfa8b3e02cb0a0b")
    card = (DATA / "triage_heldout.card.md").read_text(encoding="utf8")
    assert _sha(V1) in card and _sha(V2) in card


def test_both_versions_pass_the_structure_check():
    for path in (V1, V2):
        errors = [i for i in check_cases(load_cases(path)) if i.severity == "error"]
        assert errors == []


def test_the_test_set_has_the_composition_the_card_states():
    cases = load_cases(V2)
    counts = slice_counts(cases)
    assert len(cases) == 42
    assert counts["specialist"] == {"billing": 13, "security": 12, "technical": 17}
    assert counts["adversarial"] == {"no": 36, "yes": 6}
    assert counts["label_confidence"] == {"clear": 40, "contested": 2}
    assert all(c.split == "test" and c.canary for c in cases)


def test_no_test_case_is_a_copy_of_a_development_case():
    dev, test = load_cases(SIX), load_cases(V1)
    assert {c.case_id for c in dev}.isdisjoint({c.case_id for c in test})
    assert max(n.score for n in nearest_lexical(test, dev)) < 0.2


def test_version_two_differs_from_version_one_only_as_the_changelog_says():
    v1 = {c.case_id: c for c in load_cases(V1)}
    for c2 in load_cases(V2):
        c1 = v1[c2.case_id]
        assert c2.input == c1.input and c2.invariants == c1.invariants
        assert c2.expected["handled_by"] == c1.expected["handled_by"]
        changed = c2.expected["required_actions"] != c1.expected["required_actions"]
        assert changed == (c2.case_id in {"HO-026", "HO-041"})
        assert {
            k: v for k, v in c2.slices.items() if k != "label_confidence"
        } == c1.slices
        assert c2.slices["label_confidence"] == (
            "contested" if c2.case_id in {"HO-027", "HO-035"} else "clear"
        )


def test_the_check_catches_the_mistakes_it_exists_for():
    duplicate = [_case(), _case()]
    assert any("appears 2 times" in i.message for i in check_cases(duplicate))
    mismatch = _case(slices={"adversarial": "yes"})
    assert any("disagree" in i.message for i in check_cases([mismatch]))
    no_canary = _case(canary="")
    assert any("no canary" in i.message for i in check_cases([no_canary]))
    uneven = [_case(), _case(case_id="X-2", slices={"other": "no"})]
    assert any("slice keys" in i.message for i in check_cases(uneven))


def test_small_slices_are_warned_about_and_not_hidden():
    warnings = [i for i in check_cases(load_cases(V2)) if i.severity == "warning"]
    assert any(i.where == "adversarial=yes" for i in warnings)


def test_word_overlap_is_one_for_a_copy_and_zero_for_unrelated_text():
    assert jaccard("the app crashes on launch", "the app crashes on launch") == 1.0
    assert (
        jaccard("the app crashes on launch", "invoice total looks wrong today") == 0.0
    )


def test_the_near_dev_flag_uses_the_least_similar_known_paraphrase_as_its_line():
    semantic = [Neighbor("A", "D1", 0.9), Neighbor("B", "D2", 0.5)]
    controls = [Neighbor("p1", "D1", 0.8), Neighbor("p2", "D2", 0.7)]
    flags = near_dev_flags(semantic, controls, "model")
    assert flags["threshold"] == 0.7
    assert flags["values"] == {"A": "yes", "B": "no"}


def test_the_recorded_near_dev_slice_flags_ten_cases_including_the_injection_twin():
    flags = json.loads(
        (DATA / "triage_heldout_v1.near_dev.json").read_text(encoding="utf8")
    )
    assert Counter(flags["values"].values()) == {"no": 32, "yes": 10}
    assert flags["values"]["HO-037"] == "yes"
    assert flags["nearest"]["HO-037"]["development_case"] == "TCK-1003"


def test_the_first_run_on_the_test_set_scored_far_below_the_development_six():
    six = build_scorecard(
        "six",
        "r",
        load_cases(SIX),
        read_traces(ROOT / "runs/triage-dev-six-ch04/traces.jsonl"),
    )
    test = build_scorecard(
        "test", "r", load_cases(V1), read_traces(RUN / "traces.jsonl")
    )
    assert six.routing_successes == 6
    assert (test.routing_successes, test.observations, test.errors) == (30, 42, 12)
    assert test.invariant_violations == 0


def test_the_failures_are_the_technical_specialist_not_converging():
    traces = read_traces(RUN / "traces.jsonl")
    kinds = Counter(t.error.split(":")[0] for t in traces if t.error)
    assert kinds == {"ToolLoopDidNotConverge": 11, "HandoffLoopDetected": 1}
    cases = {c.case_id: c for c in load_cases(V1)}
    assert {cases[t.case_id].expected["handled_by"] for t in traces if t.error} == {
        "technical"
    }


def test_the_slice_table_shows_the_collapse_on_technical_tickets():
    cases, traces = load_cases(V1), read_traces(RUN / "traces.jsonl")
    table = render_by_slice(cases, traces, "specialist")
    rows = {line.split("|")[1].strip(): line for line in table.splitlines()[2:]}
    assert "13/13  77-100%" in rows["billing"]
    assert "5/17  13-53%" in rows["technical"]


def test_scoring_a_run_against_a_changed_dataset_says_so(capsys):
    base = ["stats", "--run", str(RUN)]
    main([*base, "--dataset", str(V1)])
    assert "different version of the dataset" not in capsys.readouterr().out
    main([*base, "--dataset", str(V2)])
    assert "different version of the dataset" in capsys.readouterr().out


def test_within_set_check_finds_a_planted_copy():
    a = _case(case_id="A-1")
    b = _case(case_id="A-2")
    c = _case(
        case_id="A-3",
        input={**a.input, "subject": "Other", "body": "Unrelated words here."},
    )
    pairs = within_set_pairs([a, b, c])
    assert pairs[0][1:] == ("A-1", "A-2") and pairs[0][0] == 1.0
    assert repeated_subjects([a, b, c]) == {
        a.input["subject"].strip().lower(): ["A-1", "A-2"]
    }


def test_test_set_has_no_copies_inside_it():
    cases = load_cases(V1)
    pairs = within_set_pairs(cases)
    assert len(pairs) == 861
    assert round(pairs[0][0], 2) == 0.05
    assert repeated_subjects(cases) == {}


def test_the_six_repeat_one_subject():
    repeats = repeated_subjects(load_cases(SIX))
    assert repeats == {"app crashes on startup": ["TCK-1002", "TCK-1003"]}
