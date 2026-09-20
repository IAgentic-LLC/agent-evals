"""Chapter 14: declining when the right answer is "the information does not say".

No key needed: hand-made traces, a scripted model, and the recorded runs under
runs/pkg-abstain-* and runs/triage-heldout-v1-*.
"""

import asyncio
import hashlib
import json
from pathlib import Path

import pytest
from reliable_agents_labs.rag_agent import RAG_SYSTEM_PROMPT

from agent_evals import abstention, cli, forced
from agent_evals.adapters.pkgintel import (
    PERMISSIVE_PROMPT,
    PermissivePromptAdapter,
    ScriptedRagClient,
)
from agent_evals.runner import load_cases, read_traces, run_cases
from agent_evals.schema import EvalCase, Trace

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "datasets/pkg_abstain_v1.jsonl"
ANSWERS = ROOT / "datasets/pkg_answers_v1.jsonl"
FORCED = ROOT / "datasets/triage_forced.readings.jsonl"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def has(text, *fragments):
    """Each fragment is in the text, ignoring how much white space separates words."""
    flat = " ".join(text.split())
    for fragment in fragments:
        assert " ".join(fragment.split()) in flat, fragment


def _case(kind="task", relevant=("httpx",)):
    return EvalCase(
        case_id="X-1",
        input={"query": "send an HTTP request"},
        expected={"relevant": list(relevant)},
        slices={"kind": kind},
    )


def _trace(names, cited=(), error=None, score=0.7):
    rows = [
        {"rank": i, "name": n, "summary": f"{n} summary", "score": score}
        for i, n in enumerate(names, 1)
    ]
    return Trace(
        case_id="X-1", adapter="t", retrieved=rows, cited=list(cited), error=error
    )


# ------------------------------------------------------------ the labels


def test_a_question_should_be_answered_only_when_a_relevant_package_was_retrieved():
    assert abstention.should_answer(_case(), _trace(["httpx", "requests"]))
    assert not abstention.should_answer(_case(), _trace(["requests", "urllib3"]))
    for kind in abstention.ABSTAIN_KINDS:
        assert not abstention.should_answer(_case(kind), _trace(["httpx"]))


def test_a_run_answers_when_it_cites_something_and_did_not_fail():
    assert abstention.answered(_trace(["httpx"], cited=["httpx"]))
    assert not abstention.answered(_trace(["httpx"], cited=[]))
    assert not abstention.answered(_trace(["httpx"], cited=["httpx"], error="x"))


def test_the_row_of_a_question_is_its_kind_or_retrieval_miss():
    miss = abstention.group(_case(), _trace(["requests"]))
    assert miss == "retrieval miss"
    assert abstention.group(_case(), _trace(["httpx"])) == "should answer"
    assert abstention.group(_case("outside"), _trace(["httpx"])) == "outside"


# ----------------------------------------------------------- the dataset


def test_the_abstention_set_is_the_old_questions_plus_38_new_ones():
    cases = load_cases(DATASET)
    assert len(cases) == 110
    kinds = {}
    for c in cases:
        kinds[c.slices["kind"]] = kinds.get(c.slices["kind"], 0) + 1
    assert kinds == {
        "task": 40,
        "named": 5,
        "none": 5,
        "hard": 12,
        "fresh": 10,
        "outside": 15,
        "beyond_summary": 15,
        "false_premise": 8,
    }
    old = [c for c in load_cases(ANSWERS) if c.slices["kind"] != "counterfactual"]
    assert [c.input for c in cases[:72]] == [c.input for c in old]
    assert cases[72].case_id == "AB-001"
    assert _sha(DATASET)[:8] == "969ba459"


def test_dev_and_test_alternate_within_each_kind_so_the_halves_are_balanced():
    cases = load_cases(DATASET)
    for kind in {c.slices["kind"] for c in cases}:
        splits = [c.split for c in cases if c.slices["kind"] == kind]
        assert (
            splits[0] == "dev" and abs(splits.count("dev") - splits.count("test")) <= 1
        )
    assert sum(c.split == "dev" for c in cases) == 57


# --------------------------------------------------- the prompt variant


def test_the_permissive_prompt_keeps_the_products_contract_sentences():
    for sentence in (
        "Answer the question using ONLY the package information provided below",
        "cited_packages must contain only names that appear in the provided context.",
        "leave cited_packages empty.",
        '{"answer": "your response text", "cited_packages":',
    ):
        assert sentence in RAG_SYSTEM_PROMPT and sentence in PERMISSIVE_PROMPT
    assert PERMISSIVE_PROMPT != RAG_SYSTEM_PROMPT


def test_the_adapter_patches_the_prompt_for_the_batch_and_restores_it():
    from pkgintel_app import tenant_rag

    seen = []

    class Watcher(ScriptedRagClient):
        async def generate(self, *, system, user, tools=None, history=None):
            seen.append(system)
            return await super().generate(system=system, user=user)

    case = load_cases(DATASET)[0]
    adapter = PermissivePromptAdapter(client=Watcher("faithful", []))
    asyncio.run(run_cases([case], adapter))
    assert seen == [PERMISSIVE_PROMPT]
    assert tenant_rag.RAG_SYSTEM_PROMPT == RAG_SYSTEM_PROMPT


def test_the_manifests_record_the_model_the_data_and_the_prompt():
    for name in ("pkg-abstain-1", "pkg-abstain-2"):
        m = json.loads((ROOT / "runs" / name / "manifest.json").read_text("utf8"))
        assert m["adapter"] == "pkg-live" and "prompt" not in m
        assert m["model"]["model_id"] == "gemini-3.6-flash"
        assert m["harness"]["dirty"] is False
        assert m["dataset"]["sha256"] == _sha(DATASET)
    for name in ("pkg-abstain-perm-1", "pkg-abstain-perm-2"):
        m = json.loads((ROOT / "runs" / name / "manifest.json").read_text("utf8"))
        assert m["adapter"] == "pkg-live-permissive"
        assert m["prompt"]["text"] == PERMISSIVE_PROMPT
        assert (
            m["prompt"]["sha256"]
            == hashlib.sha256(PERMISSIVE_PROMPT.encode("utf8")).hexdigest()
        )
        assert m["harness"]["dirty"] is False


# ---------------------------------------------- the recorded pkg results


@pytest.fixture(scope="module")
def recorded():
    cases = {c.case_id: c for c in load_cases(DATASET)}
    runs = {
        name: [read_traces(ROOT / "runs" / r / "traces.jsonl") for r in dirs]
        for name, dirs in abstention.CONFIGS.items()
    }
    return cases, runs


def test_over_refusal_and_wrongly_answering_for_both_prompts(recorded):
    cases, runs = recorded
    text = abstention.render_matrix(cases, runs)
    has(text, "shipped prompt       28 of 110   18-34%       0 of 110       0-3%")
    has(text, "permissive prompt    26 of 110   17-32%       2 of 110       1-6%")


def test_only_the_permissive_prompt_answered_a_question_it_should_decline(recorded):
    cases, runs = recorded
    text = abstention.render_kinds(cases, runs)
    has(text, "retrieval miss                            0 of 4             2 of 4")
    for row in (
        "outside                                  0 of 30            0 of 30",
        "beyond_summary                           0 of 30            0 of 30",
        "false_premise                            0 of 16            0 of 16",
        "none                                     0 of 10            0 of 10",
        "fresh                                    0 of 20            0 of 20",
    ):
        has(text, row)
    answered_when_wrong = [
        t.case_id
        for name, traces in runs.items()
        for one in traces
        for t in one
        if not abstention.should_answer(cases[t.case_id], t) and abstention.answered(t)
    ]
    assert answered_when_wrong == ["PQ-042", "PQ-042"]


def test_the_two_prompts_disagree_on_ten_answers_and_agree_on_the_rest(recorded):
    cases, runs = recorded
    text = abstention.render_paired(cases, runs)
    has(text, "refused under both prompts        22")
    has(text, "shipped prompt only   6")
    has(text, "permissive prompt only   4")
    has(text, "answered under both             78")


def test_the_score_cutoff_chosen_on_dev_costs_a_third_of_the_test_answers(recorded):
    cases, runs = recorded
    first = runs["shipped prompt"][0]
    assert abstention.choose_cutoff(cases, first) == pytest.approx(0.672)
    text = abstention.render_gate(cases, first)
    has(
        text,
        "0.672 21 of 28 5 of 29 22 of 27 8 of 26 <-",
        "0.600 8 of 28 0 of 29 10 of 27 0 of 26",
    )


def test_refusals_are_three_times_as_common_when_no_question_word_is_in_the_summary(
    recorded,
):
    cases, runs = recorded
    text = abstention.render_overlap(cases, runs)
    has(text, "shipped prompt            14 of 82        14 of 28")
    has(text, "permissive prompt         14 of 82        12 of 28")


def test_the_reports_fit_the_page_and_the_commands_exit_0(capsys):
    for part in ("matrix", "kinds", "paired", "scores", "gate", "overlap"):
        assert cli.main(["abstention", "--dataset", str(DATASET), "--part", part]) == 0
    for part in ("readings", "proxies"):
        assert cli.main(["forced", "--part", part]) == 0
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line


# ----------------------------------------------- the triage forced answers


@pytest.fixture(scope="module")
def forced_found():
    runs = {
        r: read_traces(ROOT / "runs" / r / "traces.jsonl")
        for group in forced.GROUPS.values()
        for r in group
    }
    return forced.forced_answers(runs), forced.load_readings(FORCED)


def test_there_are_28_forced_answers_as_chapter_10_counted(forced_found):
    found, _ = forced_found
    by_run = {}
    for run, _ in found:
        by_run[run] = by_run.get(run, 0) + 1
    assert by_run == {
        "triage-heldout-v1-calls": 4,
        "triage-heldout-v1-calls-2": 2,
        "triage-heldout-v1-guard": 12,
        "triage-heldout-v1-guard-2": 10,
    }


def test_every_forced_answer_has_a_reading_and_none_is_left_over(forced_found):
    found, readings = forced_found
    assert {(r, t.case_id) for r, t in found} == set(readings)
    assert len(readings) == 28


def test_the_guard_says_what_it_could_not_check_but_claims_sources_it_lacks(
    forced_found,
):
    found, readings = forced_found
    text = forced.render_readings(found, readings)
    has(text, "unchanged                      6      0 of 6        2 of 6")
    has(text, "stall guard                   22     22 of 22       2 of 22")
    has(text, "10 of 22")


def test_the_word_checks_agree_with_my_reading_but_not_perfectly(forced_found):
    found, readings = forced_found
    text = forced.render_proxies(found, readings)
    has(text, "gap_stated                 22       22            0       0")
    has(text, "false_action_claim          4        5            1       0")
    has(text, "claims_source_it_lacks     10       10            2       2")
