"""Chapter 15: an LLM judge, its plumbing and its reports.

No key needed: a scripted judge, hand-made items, and the recorded verdicts under
runs/judge-*.
"""

import asyncio
import hashlib
import json
from pathlib import Path

from agent_evals import judge, judge_report
from agent_evals.judge import ScriptedJudgeClient

ROOT = Path(__file__).resolve().parents[1]
ITEMS = ROOT / "datasets/judge_items_v1.jsonl"


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _item(item_id="i1", answer="pypdf merges PDFs.", label="supported"):
    return {
        "item_id": item_id,
        "group": "real",
        "kind": "real",
        "label": label,
        "reading": "supported" if label == "supported" else "stretch",
        "split": "dev",
        "clean_of_planted": False,
        "question": "merge PDFs",
        "context": [{"name": "pypdf", "summary": "A pure-python PDF library."}],
        "answer": answer,
    }


# ------------------------------------------------------------ the plumbing


def test_the_judge_sees_the_information_the_question_and_the_answer():
    text = judge.user_message(_item())
    assert text.startswith("Package information:\n- pypdf: A pure-python PDF library.")
    assert "Question: merge PDFs" in text and text.endswith(
        "Answer: pypdf merges PDFs."
    )


def test_a_scripted_judge_returns_a_verdict_and_the_tokens_it_used():
    row = asyncio.run(
        judge.judge_item(
            ScriptedJudgeClient(("best",)), "s", _item(answer="It is best.")
        )
    )
    assert row["verdict"] == "unsupported" and row["claims"][0]["supported"] is False
    assert row["input_tokens"] == 100 and row["output_tokens"] == 20


def test_a_reply_that_is_not_the_json_asked_for_is_an_error_and_not_a_guess():
    row = asyncio.run(judge.judge_item(ScriptedJudgeClient(broken=True), "s", _item()))
    assert row["verdict"] is None and "error" in row


def test_every_item_is_judged_once_per_pass():
    items = [_item("a"), _item("b")]
    rows = asyncio.run(judge.judge_all(items, ScriptedJudgeClient(), "v1", passes=2))
    assert sorted((r["item_id"], r["pass"]) for r in rows) == [
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    ]


def test_the_prompt_is_hashed_so_a_run_names_the_exact_words_it_used():
    digest = hashlib.sha256(judge.VERSIONS["v1"].encode("utf8")).hexdigest()
    assert judge.prompt_hash("v1") == digest


# ---------------------------------------------------------------- the items


def test_the_items_are_92_real_answers_and_45_planted_faults():
    items = judge.load_items(ITEMS)
    assert len(items) == 137
    real = [i for i in items if i["group"] == "real"]
    planted = [i for i in items if i["group"] == "planted"]
    assert len(real) == 92 and len(planted) == 45
    readings = {}
    for i in real:
        readings[i["reading"]] = readings.get(i["reading"], 0) + 1
    assert readings == {"supported": 85, "borderline": 6, "stretch": 1}
    kinds = {}
    for i in planted:
        kinds[i["kind"]] = kinds.get(i["kind"], 0) + 1
    assert kinds == {"praise": 15, "fact": 15, "capability": 15}
    assert sum(i["clean_of_planted"] for i in real) == 45


def test_a_planted_answer_is_its_clean_original_plus_one_added_sentence():
    items = {i["item_id"]: i for i in judge.load_items(ITEMS)}
    for i in items.values():
        if i["group"] != "planted":
            continue
        source = items[i["source_id"]]
        assert i["answer"] == f"{source['answer']} {i['added']}"
        assert i["context"] == source["context"] and i["split"] == source["split"]
        assert source["clean_of_planted"] and source["reading"] == "supported"


def test_no_added_sentence_names_a_package_or_a_number_so_the_code_checks_cannot_see_it():
    from agent_evals import grounding
    from agent_evals.retrieval import load_corpus

    names = [r["name"] for r in load_corpus(ROOT / "datasets/pkg_corpus_v1.jsonl")]
    for i in judge.load_items(ITEMS):
        if i["group"] == "planted":
            assert not grounding.VERSION.findall(i["added"])
            assert not [n for n in names if grounding.mentions(i["added"], n)]


def test_the_two_halves_hold_every_kind_of_planted_fault():
    items = judge.load_items(ITEMS)
    for split in ("dev", "test"):
        kinds = {
            i["kind"] for i in items if i["group"] == "planted" and i["split"] == split
        }
        assert kinds == {"praise", "fact", "capability"}


# --------------------------------------------------------------- the reports


def _rows(items, said):
    return [
        {
            "item_id": i["item_id"],
            "pass": p,
            "verdict": "unsupported" if i["item_id"] in said else "supported",
            "claims": [{"claim": "c", "supported": False}],
            "seconds": 1.0,
            "input_tokens": 10,
            "output_tokens": 2,
        }
        for i in items
        for p in (1, 2)
    ]


def test_the_planted_report_counts_faults_found_and_clean_answers_flagged():
    items = judge.load_items(ITEMS)
    faulty = {
        i["item_id"] for i in items if i["group"] == "planted" and i["kind"] == "fact"
    }
    text = judge_report.render_planted(items, _rows(items, faulty))
    flat = " ".join(text.split())
    assert "with fact added 30 of 30" in flat
    assert "with praise added 0 of 30" in flat
    assert "the same answers, clean 0 of 90" in flat


def test_the_real_report_sets_the_judge_against_my_readings_and_lists_disagreements():
    items = judge.load_items(ITEMS)
    stretch = next(i for i in items if i["reading"] == "stretch")
    wrong = next(i for i in items if i["reading"] == "supported")
    rows = _rows(items, {stretch["item_id"], wrong["item_id"]})
    flat = " ".join(judge_report.render_real(items, rows).split())
    assert "stretch 2 of 2" in flat and "supported 2 of 170" in flat
    found = judge_report.disagreements(items, rows)
    assert [i["item_id"] for i, _ in found] == [wrong["item_id"]]


def test_test_retest_counts_items_with_the_same_verdict_in_both_passes():
    items = [_item("a"), _item("b")]
    rows = _rows(items, {"a"})
    rows[3]["verdict"] = "unsupported"  # item b, second pass
    assert "1 of 2" in judge_report.render_retest(items, rows)


def test_the_cost_report_totals_tokens_and_counts_replies_that_were_not_json():
    rows = _rows([_item("a")], set())
    rows[1]["verdict"] = None
    text = judge_report.render_cost(rows)
    assert "calls" in text and "input tokens" in text
    assert " ".join(text.split()).startswith("calls 2 replies that were not JSON 1")


# ----------------------------------------------------------- recorded runs

I = str(ITEMS)


def _has(text, *fragments):
    flat = " ".join(text.split())
    for fragment in fragments:
        assert " ".join(fragment.split()) in flat, fragment


def _rows_of(*runs):
    return judge_report.load_rows([ROOT / "runs" / r for r in runs])


def test_the_recorded_runs_name_their_model_prompt_items_and_a_clean_commit():
    for name, version, split, count in (
        ("judge-v1", "v1", None, 137),
        ("judge-v2-test", "v2", "test", 64),
    ):
        m = json.loads((ROOT / "runs" / name / "manifest.json").read_text("utf8"))
        assert m["judge_model"]["model_id"] == "gemini-3.8-flash"
        assert m["prompt"]["version"] == version
        assert m["prompt"]["sha256"] == judge.prompt_hash(version)
        assert m["prompt"]["text"] == judge.VERSIONS[version]
        assert m["items"]["sha256"] == _sha(ITEMS)
        assert (m["items"]["count"], m["items"]["split"]) == (count, split)
        assert m["passes"] == 2 and m["harness"]["dirty"] is False


def test_version_2_differs_from_version_1_only_by_the_added_instruction():
    assert judge.VERSIONS["v2"].startswith(judge.VERSIONS["v1"])
    assert judge.VERSIONS["v2"] != judge.VERSIONS["v1"]


def test_version_1_finds_every_planted_fault_and_flags_over_a_third_of_clean_answers():
    items = judge.load_items(ITEMS)
    text = judge_report.render_planted(items, _rows_of("judge-v1"))
    _has(
        text,
        "with praise added 30 of 30",
        "with fact added 30 of 30",
        "with capability added 30 of 30",
        "the same answers, clean 37 of 90",
    )


def test_version_1_agrees_with_my_borderline_and_stretch_readings_and_flags_many_supported():
    items = judge.load_items(ITEMS)
    text = judge_report.render_real(items, _rows_of("judge-v1"))
    _has(
        text,
        "supported 59 of 170",
        "borderline 12 of 12",
        "stretch 2 of 2",
    )


def test_two_passes_give_the_same_verdict_on_nearly_every_item():
    items = judge.load_items(ITEMS)
    _has(judge_report.render_retest(items, _rows_of("judge-v1")), "134 of 137")
    text = judge_report.render_retest(items, _rows_of("judge-v2-test"), "test")
    _has(text, "64 of 64")


def test_on_the_test_half_version_2_keeps_every_fault_and_drops_the_false_alarms():
    items = judge.load_items(ITEMS)
    text = judge_report.render_compare(
        items,
        judge_report.load_rows([ROOT / "runs/judge-v1"]),
        judge_report.load_rows([ROOT / "runs/judge-v2-test"]),
        ["judge-v1", "judge-v2-test"],
        "test",
    )
    _has(
        text,
        "with praise added 12 of 12 12 of 12",
        "with fact added 12 of 12 12 of 12",
        "with capability added 12 of 12 12 of 12",
        "planted originals, clean 16 of 36 2 of 36",
        "real, my reading: supported 24 of 80 2 of 80",
        "real, my reading: borderline 12 of 12 4 of 12",
    )


def test_the_one_supported_answer_version_2_still_flags_in_every_pass_on_test():
    items = judge.load_items(ITEMS)
    found = judge_report.disagreements(items, _rows_of("judge-v2-test"), "test")
    assert [i["item_id"] for i, _ in found] == ["pkg-answers-1:PQ-013"]
    assert "SQLAlchemy to connect to a Postgres database" in found[0][1]


def test_the_cost_report_totals_tokens_and_prices_only_when_given_prices():
    text = judge_report.render_cost(_rows_of("judge-v1"))
    _has(text, "calls 274", "input tokens 69470", "output tokens 17296")
    assert "dollars" not in text
    priced = judge_report.render_cost(_rows_of("judge-v1"), 0.75, 3.75)
    _has(priced, "dollars at the given prices 0.117", "dollars per judgment 0.00043")


def test_the_report_commands_fit_the_page_and_exit_0(capsys):
    from agent_evals import cli

    for part in ("planted", "real", "retest", "cost"):
        assert (
            cli.main(
                [
                    "judge",
                    "report",
                    "--run",
                    "runs/judge-v1",
                    "--items",
                    I,
                    "--part",
                    part,
                ]
            )
            == 0
        )
    both = ["--run", "runs/judge-v1", "runs/judge-v2-test", "--items", I]
    assert (
        cli.main(["judge", "report", *both, "--part", "compare", "--split", "test"])
        == 0
    )
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line
