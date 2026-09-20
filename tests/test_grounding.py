"""Chapter 13: answers checked against what was retrieved.

No key needed: hand-made traces, a scripted model over the recorded embeddings, and the
recorded runs under runs/pkg-answers-*.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest

from agent_evals import cli, grounding, grounding_check
from agent_evals.adapters.pkgintel import PkgAnswerAdapter, ScriptedRagClient
from agent_evals.retrieval import load_corpus
from agent_evals.runner import load_cases, read_traces, run_cases
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/pkg_answers_v1.jsonl"
V2 = ROOT / "datasets/pkg_queries_v2.jsonl"
READINGS = ROOT / "datasets/pkg_answers_v1.readings.jsonl"
RUNS = [ROOT / "runs/pkg-answers-1", ROOT / "runs/pkg-answers-2"]
NAMES = [r["name"] for r in load_corpus(ROOT / "datasets/pkg_corpus_v1.jsonl")]


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case(kind="task", relevant=("httpx",), query="send an HTTP request", **expected):
    return EvalCase(
        case_id="X-1",
        input={"query": query},
        expected={"relevant": list(relevant), **expected},
        slices={"kind": kind},
    )


def _trace(answer, cited, retrieved=None, error=None):
    retrieved = retrieved or [
        {"rank": 1, "name": "httpx", "summary": "The next generation HTTP client."},
        {"rank": 2, "name": "requests", "summary": "Python HTTP for Humans."},
    ]
    return Trace(
        case_id="X-1",
        adapter="t",
        answer=answer,
        cited=cited,
        retrieved=retrieved,
        error=error,
    )


# ------------------------------------------------------------ the checks


def test_a_name_is_matched_as_a_whole_word_and_ignoring_case():
    assert grounding.mentions("Try Rich for this", "rich")
    assert not grounding.mentions("which one", "rich")
    assert grounding.mentions("use python-dotenv here", "python-dotenv")
    assert not grounding.mentions("use python-dotenv-extra", "python-dotenv")


def test_a_citation_of_a_package_that_was_not_retrieved_is_flagged():
    found = grounding.check(_case(), _trace("httpx.", ["httpx", "numpy"]), NAMES)
    assert found["invalid_citation"] is True
    found = grounding.check(_case(), _trace("httpx.", ["HTTPX"]), NAMES)
    assert found["invalid_citation"] is False  # case is not the point


def test_a_package_named_outside_the_context_is_flagged_and_one_inside_is_not():
    outside = grounding.check(_case(), _trace("Try httpx or numpy.", ["httpx"]), NAMES)
    assert outside["outside"] == ["numpy"]
    inside = grounding.check(_case(), _trace("httpx beats requests.", ["httpx"]), NAMES)
    assert inside["outside_name"] is False  # both were retrieved
    asked = _case(query="is numpy any good?")
    named = grounding.check(asked, _trace("numpy is out of scope.", []), NAMES)
    assert named["outside_name"] is False  # the person named it


def test_a_number_the_context_does_not_hold_is_flagged():
    flagged = grounding.check(_case(), _trace("It is 2.31.0 now.", ["httpx"]), NAMES)
    assert flagged["stated"] == ["2.31.0"] and flagged["new_number"] is True
    said = "Needs Python 4.0."
    rows = [{"rank": 1, "name": "pandas", "summary": "Needs Python 4.0 or newer."}]
    same = grounding.check(_case(), _trace(said, ["pandas"], rows), NAMES)
    assert same["new_number"] is False


def test_no_citation_applies_only_when_a_relevant_package_was_retrieved():
    quiet = _trace("I cannot say.", [])
    assert grounding.check(_case(), quiet, NAMES)["no_citation"] is True
    assert grounding.check(_case(relevant=()), quiet, NAMES)["no_citation"] is None
    other = _case(relevant=("fastapi",))
    assert grounding.check(other, quiet, NAMES)["no_citation"] is None
    fresh = _case(kind="fresh")
    assert grounding.check(fresh, quiet, NAMES)["no_citation"] is None


def test_the_counterfactual_check_looks_for_the_added_fact():
    case = _case(kind="counterfactual", markers=["apache"])
    yes = grounding.check(case, _trace("Under Apache-2.0.", ["httpx"]), NAMES)
    no = grounding.check(case, _trace("Under BSD.", ["httpx"]), NAMES)
    assert (yes["ignored_context"], no["ignored_context"]) == (False, True)
    assert (
        grounding.check(_case(), _trace("x", ["httpx"]), NAMES)["ignored_context"]
        is None
    )


def test_a_failed_run_is_an_error_and_nothing_else():
    found = grounding.check(_case(), _trace("", [], error="ValueError: x"), NAMES)
    assert found["error"] is True and found["invalid_citation"] is None


# --------------------------------------------------------------- the dataset


def test_the_answer_set_is_the_62_queries_then_10_fresh_and_5_counterfactual():
    cases = load_cases(DATASET)
    assert len(cases) == 77
    kinds = [c.slices["kind"] for c in cases]
    assert kinds.count("fresh") == 10 and kinds.count("counterfactual") == 5
    assert kinds[:62] == [c.slices["kind"] for c in load_cases(V2)]
    first = DATASET.read_text(encoding="utf8").splitlines()[:62]
    assert first == V2.read_text(encoding="utf8").splitlines()
    assert _sha(DATASET)[:8] == "d7c726cc"
    corpus = {r["name"]: r for r in load_corpus(ROOT / "datasets/pkg_corpus_v1.jsonl")}
    for case in cases:
        if case.slices["kind"] == "fresh":
            name = case.expected["relevant"][0]
            assert case.expected["version"] == corpus[name]["version"]
        if case.slices["kind"] == "counterfactual":
            ((name, text),) = case.input["summary_edits"].items()
            assert (
                text.startswith(corpus[name]["summary"])
                and text != corpus[name]["summary"]
            )


def _scripted(mode="faithful"):
    return PkgAnswerAdapter(client=ScriptedRagClient(mode, NAMES))


def test_retrieval_for_the_new_questions_is_what_the_dataset_relies_on():
    cases = load_cases(DATASET)
    traces = asyncio.run(run_cases(cases, _scripted(), concurrency=1))
    by = {t.case_id: t for t in traces}
    for case in cases:
        got = [r["name"] for r in by[case.case_id].retrieved]
        if case.slices["kind"] == "counterfactual":
            assert case.expected["relevant"][0] in got  # the edited package is seen
            edited = next(iter(case.input["summary_edits"].values()))
            assert edited in [r["summary"] for r in by[case.case_id].retrieved]
    missed = [
        c.case_id
        for c in cases
        if c.slices["kind"] == "fresh"
        and c.expected["relevant"][0]
        not in [r["name"] for r in by[c.case_id].retrieved]
    ]
    assert missed == [
        "FQ-005",
        "FQ-006",
    ]  # pandas and pydantic, whose names are not embedded


def test_the_grounding_check_passes_and_shows_its_blind_spot(capsys):
    assert cli.main(["grounding", "check"]) == 0
    out = capsys.readouterr().out
    assert "faithful           (none)                     -            0" in out
    assert "cites_nothing      no_citation         60 of 60            0" in out
    assert "adds an unsupported claim: flagged in 0 of 77 answers" in out


def test_a_faulty_script_that_the_checks_miss_would_fail_the_command(
    monkeypatch, capsys
):
    monkeypatch.setitem(grounding_check.FAULTS, "adds_claim", "invalid_citation")
    assert cli.main(["grounding", "check"]) == 1


# ------------------------------------------------------- the recorded answers


@pytest.fixture(scope="module")
def recorded():
    cases = {c.case_id: c for c in load_cases(DATASET)}
    runs = {r.name: read_traces(r / "traces.jsonl") for r in RUNS}
    return cases, runs


def test_the_manifests_name_the_model_the_data_and_a_clean_commit():
    for run in RUNS:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf8"))
        assert manifest["adapter"] == "pkg-live"
        assert manifest["model"]["model_id"] == "gemini-3.6-flash"
        assert manifest["dataset"]["sha256"] == _sha(DATASET)
        assert manifest["harness"]["dirty"] is False
        assert "pkgintel-app" in manifest["packages"]


def test_the_recorded_answers_pass_every_check_except_the_cost_of_refusing(recorded):
    cases, runs = recorded
    traces = [t for ts in runs.values() for t in ts]
    table = grounding.tally(cases, traces, NAMES)["all"]
    assert table["runs"] == 154 and table["error"] == 0
    for flag in ("invalid_citation", "outside_name", "new_number", "ignored_context"):
        assert table[flag] == 0
    assert (table["no_citation"], table["no_citation_of"]) == (29, 120)
    assert (table["invalid_citation_of"], table["ignored_context_of"]) == (154, 10)


def test_every_fresh_answer_states_no_version_and_the_model_follows_edited_context(
    recorded,
):
    cases, runs = recorded
    for traces in runs.values():
        for t in traces:
            kind = cases[t.case_id].slices["kind"]
            if kind == "fresh":
                assert grounding.VERSION.findall(t.answer) == []
            if kind == "counterfactual":
                assert (
                    grounding.check(cases[t.case_id], t, NAMES)["ignored_context"]
                    is False
                )


def test_the_model_refuses_a_quarter_of_answerable_questions_it_could_answer(recorded):
    cases, runs = recorded
    text = grounding.render_refusals(cases, runs)
    assert "pkg-answers-1                         15 of 55" in text
    assert "pkg-answers-2                         14 of 55" in text
    assert "all runs                             29 of 110" in text
    assert (
        "55 questions: answered in every run 39, refused in every run 13, mixed 3"
        in text
    )


def test_the_hand_readings_cover_every_answer_and_agree_with_the_traces(recorded):
    cases, runs = recorded
    readings = grounding.load_readings(READINGS)
    assert len(readings) == 154
    key = {(r["run"], r["case_id"]): r["reading"] for r in readings}
    assert len(key) == 154
    for run, traces in runs.items():
        for t in traces:
            assert (key[(run, t.case_id)] == "refusal") == (not t.cited)
    counts = {k: sum(v == k for v in key.values()) for k in set(key.values())}
    assert counts == {
        "supported": 85,
        "refusal": 62,
        "borderline": 6,
        "stretch": 1,
    }
    assert key[("pkg-answers-1", "PQ-042")] == "stretch"
    beyond = grounding.flagged(
        cases, [t for ts in runs.values() for t in ts], NAMES, grounding.BEYOND
    )
    assert beyond == []  # the one stretch was not seen by any check


def test_the_reports_fit_the_page_and_the_command_exits_0(recorded, capsys):
    base = ["grounding", "grade", "--run", *map(str, RUNS), "--dataset", str(DATASET)]
    for part in (
        "summary",
        "bounds",
        "fresh",
        "counterfactual",
        "readings",
        "refusals",
    ):
        assert cli.main([*base, "--part", part]) == 0
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line


def test_an_invalid_citation_in_a_recorded_run_fails_the_command(tmp_path):
    traces = read_traces(RUNS[0] / "traces.jsonl")
    traces[0] = traces[0].model_copy(update={"cited": [*traces[0].cited, "numpy-x"]})
    (tmp_path / "traces.jsonl").write_text(
        "".join(t.model_dump_json() + "\n" for t in traces), encoding="utf8"
    )
    args = ["grounding", "grade", "--run", str(tmp_path), "--dataset", str(DATASET)]
    assert cli.main(args) == 1


def test_from_memory_the_model_gives_stale_versions_for_nine_of_ten_packages():
    probe = ROOT / "runs/pkg-memory-probe/answers.jsonl"
    rows = [json.loads(x) for x in probe.read_text(encoding="utf8").splitlines()]
    truth = {
        c.case_id: c.expected["version"]
        for c in load_cases(DATASET)
        if c.expected.get("version")
    }
    stated = {r["case_id"]: set() for r in rows}
    for r in rows:
        stated[r["case_id"]] |= set(grounding.VERSION.findall(r["answer"]))
    gave = [c for c, v in stated.items() if v]
    right = [c for c in gave if truth[c] in stated[c]]
    assert len(gave) == 9 and right == ["FQ-002"]
