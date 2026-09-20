"""Chapter 17: agreement between raters, and what an imperfect judge does to a rate.

The kappa values below were computed once with scikit-learn's `cohen_kappa_score` and are
pinned here, so this code is checked against an established implementation.
"""

import json
from pathlib import Path

import pytest

from agent_evals import agreement, cli, labels

ROOT = Path(__file__).resolve().parents[1]


def _has(text, *fragments):
    flat = " ".join(text.split())
    for fragment in fragments:
        assert " ".join(fragment.split()) in flat, fragment


# ------------------------------------------------------------------- the metrics


def test_kappa_matches_scikit_learn_on_two_pinned_cases():
    a = list("aabbbaabcabcabbcaabbccab")
    b = list("aabbcaabbabcbbbcaabcccab")
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.7433155080213905, abs=1e-12)
    a2 = ["y"] * 40 + ["n"] * 10
    b2 = ["y"] * 38 + ["n"] * 2 + ["y"] * 3 + ["n"] * 7
    assert agreement.cohen_kappa(a2, b2) == pytest.approx(0.6753246753246753, abs=1e-12)


def test_kappa_on_the_textbook_example_is_0_4():
    a = ["y"] * 25 + ["n"] * 25
    b = ["y"] * 20 + ["n"] * 5 + ["y"] * 10 + ["n"] * 15
    assert agreement.cohen_kappa(a, b) == pytest.approx(0.4)


def test_kappa_is_1_for_identical_labels_and_0_for_agreement_no_better_than_chance():
    a = ["y", "n"] * 10
    assert agreement.cohen_kappa(a, a) == 1.0
    assert agreement.cohen_kappa(["y", "y", "n", "n"], ["y", "n", "y", "n"]) == 0.0
    assert agreement.cohen_kappa(["y"] * 5, ["y"] * 5) == 1.0


def test_pabak_does_not_shrink_when_one_label_is_rare_but_kappa_does():
    a = ["ok"] * 95 + ["bad"] * 5
    b = ["ok"] * 93 + ["bad"] * 2 + ["ok"] * 3 + ["bad"] * 2
    assert agreement.percent_agreement(a, b) == pytest.approx(0.95)
    assert agreement.pabak(a, b) == pytest.approx(0.90)
    assert agreement.cohen_kappa(a, b) < 0.5


def test_the_bootstrap_interval_is_reproducible_and_wider_when_answers_share_a_question():
    pairs = [("y", "y"), ("n", "n"), ("y", "n"), ("n", "y")] * 3
    pairs = pairs[:6] + [("y", "y")] * 3 + [("n", "n")] * 3
    a = [x for x, _ in pairs for _ in range(4)]
    b = [y for _, y in pairs for _ in range(4)]
    solo = agreement.bootstrap_interval(agreement.cohen_kappa, a, b, resamples=1500)
    assert solo == agreement.bootstrap_interval(
        agreement.cohen_kappa, a, b, resamples=1500
    )
    group = [str(i // 4) for i in range(len(a))]
    grouped = agreement.bootstrap_interval(
        agreement.cohen_kappa, a, b, groups=group, resamples=1500
    )
    assert grouped[1] - grouped[0] > solo[1] - solo[0]


# ------------------------------------------------------- correcting a judge's rate


def test_rogan_gladen_recovers_the_true_rate_from_a_flag_rate():
    assert agreement.rogan_gladen(0.2, 0.9, 0.95) == pytest.approx(0.15 / 0.85)
    assert agreement.rogan_gladen(0.0, 0.9, 0.95) == 0.0  # clipped, never negative
    assert agreement.rogan_gladen(1.0, 0.9, 0.95) == 1.0
    with pytest.raises(ZeroDivisionError):
        agreement.rogan_gladen(0.3, 0.5, 0.5)


def test_a_perfect_judge_needs_no_correction():
    truth = [True] * 10 + [False] * 30
    groups = [str(i) for i in range(40)]
    point, (low, high) = agreement.calibrated_rate(truth, truth, groups, truth, groups)
    assert point == pytest.approx(0.25) and low <= 0.25 <= high


def test_sensitivity_and_specificity_count_the_right_cells():
    truth = [True, True, True, False, False, False, False]
    flags = [True, True, False, True, False, False, False]
    sens, spec = agreement.sensitivity_specificity(truth, flags)
    assert sens == pytest.approx(2 / 3) and spec == pytest.approx(3 / 4)


def test_a_question_always_lands_in_the_same_half():
    assert agreement.group_of("connect to a Postgres database") == agreement.group_of(
        "connect to a Postgres database"
    )
    assert {agreement.group_of(str(i)) for i in range(50)} == {0, 1}


def test_more_calibration_items_narrow_the_interval_and_the_test_set_sets_the_floor():
    small = dict(agreement.planning_widths(0.85, 0.97, 0.08, 100, (50, 800)))
    big = dict(agreement.planning_widths(0.85, 0.97, 0.08, 400, (50, 800)))
    assert small[800] < small[50] and big[800] < big[50]
    assert big[800] < small[800]


# --------------------------------------------------------------- the recorded labels

JUDGES = [
    "judge-v1",
    "judge-v2-all",
    "judge-v3-all",
    "judge-v3u-all",
    "judge-v2-self",
    "judge-v2-other",
]


def _items():
    return labels.item_list()


def _matrix(adjudicated=False, convention="strict", group="real"):
    items = _items()
    truth = labels.author_labels(items, convention, adjudicated)
    raters = {r.replace("judge-", ""): labels.judge_labels(r) for r in JUDGES}
    return labels.render_matrix(items, raters, truth, group)


def test_my_labels_have_7_bad_real_answers_and_9_after_adjudication():
    items = [i for i in _items() if i["group"] == "real"]
    count = lambda t: sum(v == "unsupported" for v in t.values())
    assert count(labels.author_labels(items)) == 7
    assert count(labels.author_labels(items, adjudicated=True)) == 9
    assert count(labels.author_labels(items, "lenient")) == 1


def test_changing_two_labels_of_92_moves_kappa_a_great_deal():
    before, after = _matrix(), _matrix(adjudicated=True)
    _has(before, "v2-all 92 0.924 0.33 (-0.06 to 0.82) 0.85")
    _has(after, "v2-all 92 0.946 0.59 (0.00 to 1.00) 0.89")
    _has(before, "v3-all 92 0.924 0.42 (-0.05 to 0.82) 0.85")
    _has(after, "v3-all 92 0.924 0.49 (-0.02 to 0.85) 0.85")


def test_the_convention_for_borderline_changes_the_agreement():
    _has(_matrix(convention="lenient"), "v2-all 92 0.946 -0.02 (-0.04 to 1.00) 0.89")


def test_on_all_items_the_planted_faults_make_agreement_look_high():
    text = _matrix(group="all")
    _has(
        text,
        "v2-all 137 0.949 0.89 (0.77 to 0.98) 0.90",
        "v1 137 0.788 0.59 (0.46 to 0.73) 0.58",
    )


def test_a_judge_agrees_with_itself_on_nearly_every_item():
    text = labels.render_retest(_items(), JUDGES)
    _has(
        text,
        "v2-all 137 1.000 1.00",
        "v3-all 137 0.971 0.94",
        "v2-other 137 0.978 0.95",
    )


def test_sensitivity_from_seven_bad_answers_in_four_questions_is_almost_unknown():
    items = _items()
    truth = labels.author_labels(items)
    text = labels.render_sensitivity(items, truth, labels.judge_labels("judge-v3-all"))
    _has(text, "sensitivity 0.43 0.00 to 1.00", "bad answers 7 of 92, from 4 questions")


def test_the_calibration_half_holds_one_bad_answer_and_the_correction_goes_wrong():
    items = _items()
    truth = labels.author_labels(items)
    text = labels.render_calibration(
        items, truth, labels.judge_labels("judge-v3-all"), "v3-all"
    )
    _has(
        text,
        "calibration questions 43 1 4",
        "evaluation questions 49 6 2",
        "sensitivity 1.00 and specificity 0.93",
        "my labels give on the evaluation questions: 12.2%",
        "corrected rate: 0.0% (0.0% to 8.2%)",
    )
    v2 = labels.render_calibration(
        items, truth, labels.judge_labels("judge-v2-all"), "v2-all"
    )
    _has(v2, "no correction possible")


def test_the_plan_table_flattens_when_the_test_set_is_small():
    text = labels.render_plan(0.85, 0.97, 0.08, (50, 100, 200, 400, 800), (100, 400))
    _has(text, "50 22.0 pts 19.2 pts", "800 14.9 pts 7.9 pts")


# ------------------------------------------------------- the second person's page


def test_the_sample_is_50_answers_with_the_hard_ones_in_it():
    rows = [
        json.loads(x)
        for x in (ROOT / "datasets/human_label_sample_v1.jsonl")
        .read_text("utf8")
        .splitlines()
    ]
    assert len(rows) == 50 and len({r["item_id"] for r in rows}) == 50
    strata = {}
    for r in rows:
        strata[r["stratum"]] = strata.get(r["stratum"], 0) + 1
    assert strata == {
        "author: borderline or stretch": 7,
        "author: supported, a judge disagrees": 4,
        "author: supported": 31,
        "planted fault": 8,
    }


def test_the_page_shows_the_question_the_information_and_the_answer_and_nothing_else():
    page = (ROOT / "label-tool/index.html").read_text(encoding="utf8")
    items = json.loads(
        page.split('id="items">')[1].split("</script>")[0].replace("<\\/", "</")
    )
    assert len(items) == 50 and set(items[0]) == {"id", "question", "context", "answer"}
    for word in (
        "stratum",
        "borderline",
        "judge disagrees",
    ):
        assert word not in json.dumps(items)


def test_a_second_persons_labels_are_compared_with_mine_and_a_judges(tmp_path, capsys):
    items = _items()
    mine = labels.author_labels(items)
    sample = [
        json.loads(x)["item_id"]
        for x in (ROOT / "datasets/human_label_sample_v1.jsonl")
        .read_text("utf8")
        .splitlines()
    ]
    rows = [
        {"item_id": i, "label": mine[i], "note": "", "labeler": "t"} for i in sample
    ]
    rows[0]["label"] = "unsure"
    path = tmp_path / "second.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf8")
    args = [
        "labels",
        "--part",
        "second",
        "--judge",
        "judge-v3-all",
        "--second",
        str(path),
    ]
    assert cli.main(args) == 0
    out = capsys.readouterr().out
    _has(out, "49 labeled, 1 marked not sure", "author 49 1.000 1.00")


def test_the_adjudication_file_records_seven_decisions_and_two_changes():
    rows = [
        json.loads(x)
        for x in (ROOT / "datasets/judge_adjudication_v1.jsonl")
        .read_text("utf8")
        .splitlines()
    ]
    assert len(rows) == 7 and sum(r["was"] != r["decision"] for r in rows) == 2
    assert {r["item_id"] for r in rows if r["was"] != r["decision"]} == {
        "pkg-answers-1:PQ-013",
        "pkg-answers-2:PQ-013",
    }


def test_the_report_commands_fit_the_page(capsys):
    j = ["--judge", "judge-v2-all", "judge-v3-all"]
    for part in ("matrix", "retest"):
        assert cli.main(["labels", "--part", part, *j]) == 0
    for part in ("sensitivity", "calibrate"):
        assert cli.main(["labels", "--part", part, "--judge", "judge-v3-all"]) == 0
    assert cli.main(["labels", "--part", "plan"]) == 0
    for line in capsys.readouterr().out.splitlines():
        assert len(line) <= 78, line
