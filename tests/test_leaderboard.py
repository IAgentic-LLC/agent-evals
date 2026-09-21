"""Chapter 26: the leaderboard checked against runs whose truth is known.

The board is an evaluator, so it is tested the way the book tests every evaluator: feed it
entries built to be good, bad, tied and cheating, and check that it says so. The tables
for the real board are pinned on the recorded runs.
"""

import contextlib
import hashlib
import io
import json
from pathlib import Path

import yaml

from agent_evals import cli, leaderboard
from agent_evals.runner import load_cases, write_traces
from agent_evals.schema import Trace

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "datasets/triage_incident_v1.jsonl"
PRICES = ROOT / "config/prices.yaml"


def _dataset(tmp_path, tickets=6):
    lines = SOURCE.read_text(encoding="utf8").splitlines()[:tickets]
    path = tmp_path / "datasets" / "small.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf8", newline="\n")
    return path


def _run(
    tmp_path,
    name,
    dataset,
    good=1.0,
    trials=3,
    dirty=False,
    model="gemini-3.6-flash",
    metered=True,
    other_customer=False,
    leftover=0,
    drop_manifest_field=None,
    manifest=True,
    sha=None,
):
    """Write a run folder. `good` is the share of trials that take the required action,
    which is the first `good * trials` trials of every ticket."""
    cases = load_cases(dataset)
    traces = []
    for case in cases:
        own = case.input["customer_id"]
        for k in range(1, trials + 1):
            ok = k <= round(good * trials)
            acts = [{"action": "look_up_invoice", "customer_id": own}] if ok else []
            if other_customer and k == 1:
                acts.append({"action": "look_up_invoice", "customer_id": "cust_other"})
            usage = {"model_calls": 2, "input_tokens": 2000, "output_tokens": 300}
            if metered:
                usage["thinking_tokens"] = 100
            traces.append(
                Trace(
                    case_id=case.case_id,
                    trial=k,
                    adapter="test",
                    handled_by="billing",
                    actions_taken=acts,
                    latency_s=1.0 + k,
                    usage=usage,
                    ledger_at_start=leftover if k == 1 else 0,
                )
            )
    run = tmp_path / "runs" / name
    write_traces(run / "traces.jsonl", traces)
    if manifest:
        data = {
            "adapter": "test",
            "model": {"provider": "gemini", "model_id": model},
            "dataset": {
                "path": "datasets/small.jsonl",
                "sha256": sha or hashlib.sha256(dataset.read_bytes()).hexdigest(),
                "cases": len(cases),
            },
            "trials": trials,
            "harness": {"commit": "abc1234", "dirty": dirty},
        }
        if drop_manifest_field:
            data.pop(drop_manifest_field)
        (run / "manifest.json").write_text(json.dumps(data), encoding="utf8")
    return run


def _board(tmp_path, entries, min_trials=3):
    spec = {
        "name": "test board",
        "dataset": "datasets/small.jsonl",
        "min_trials": min_trials,
        "prices": str(PRICES),
        "entries": [{"run": r, "label": label} for r, label in entries],
        "first_look": [label for _, label in entries],
    }
    path = tmp_path / "board.yaml"
    path.write_text(yaml.safe_dump(spec), encoding="utf8")
    return leaderboard.load_board(path)


def _build(tmp_path, board):
    # prices are read from an absolute path, so root can be the temporary folder
    return leaderboard.build(tmp_path, board)


# --------------------------------------------------------------------- admission


def test_a_clean_run_with_enough_trials_is_admitted(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "clean", ds)
    rows, refused, _ = _build(tmp_path, _board(tmp_path, [("clean", "clean")]))
    assert len(rows) == 1 and refused == []


def test_each_thing_wrong_with_a_run_is_refused_for_its_own_reason(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "no-manifest", ds, manifest=False)
    _run(tmp_path, "missing", ds, drop_manifest_field="harness")
    _run(tmp_path, "dirty", ds, dirty=True)
    _run(tmp_path, "short", ds, trials=1)
    _run(tmp_path, "other", ds, sha="f" * 64)
    _run(tmp_path, "leftover", ds, leftover=2)
    names = ["no-manifest", "missing", "dirty", "short", "other", "leftover"]
    _, refused, _ = _build(tmp_path, _board(tmp_path, [(n, n) for n in names]))
    why = {e.run: reasons for e, reasons in refused}
    assert why["no-manifest"] == ["no manifest"]
    assert why["missing"] == ["manifest lacks harness.commit, harness.dirty"]
    assert why["dirty"] == ["harness had uncommitted changes"]
    assert why["short"] == ["1 trial, the board needs 3"]
    assert why["other"][0].startswith("another dataset (ffffff")
    assert why["leftover"] == ["6 runs began with leftover state"]


def test_a_run_missing_some_tickets_is_refused(tmp_path):
    ds = _dataset(tmp_path)
    run = _run(tmp_path, "partial", ds)
    traces = [
        json.loads(line)
        for line in (run / "traces.jsonl").read_text(encoding="utf8").splitlines()
    ]
    keep = [t for t in traces if t["case_id"] != traces[0]["case_id"]]
    (run / "traces.jsonl").write_text(
        "\n".join(json.dumps(t) for t in keep) + "\n", encoding="utf8"
    )
    _, refused, _ = _build(tmp_path, _board(tmp_path, [("partial", "partial")]))
    assert refused[0][1] == ["1 tickets have no runs"]


# -------------------------------------------------------------------------- rows


def test_an_entry_that_acted_on_another_customer_is_blocked_and_never_ranked(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "cheat", ds, good=1.0, other_customer=True)
    _run(tmp_path, "honest", ds, good=2 / 3)
    board = _board(tmp_path, [("cheat", "cheat"), ("honest", "honest")])
    rows, _, _ = _build(tmp_path, board)
    text = leaderboard.render_board(rows)
    ranked_part = text.split("BLOCKED")[0]
    assert "honest" in ranked_part and "cheat" not in ranked_part
    assert "cheat" in text.split("BLOCKED")[1]
    assert next(r for r in rows if r.label == "cheat").customer == 6


def test_cost_is_shown_only_for_a_metered_run(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "metered", ds)
    _run(tmp_path, "unmetered", ds, metered=False)
    board = _board(tmp_path, [("metered", "metered"), ("unmetered", "unmetered")])
    rows, _, _ = _build(tmp_path, board)
    by = {r.label: r for r in rows}
    assert by["metered"].cost_per_success is not None
    assert by["unmetered"].cost_per_success is None
    assert "n/a" in leaderboard.render_board(rows)


def test_steady_is_the_share_of_tickets_met_in_every_trial(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "twothirds", ds, good=2 / 3)
    rows, _, _ = _build(tmp_path, _board(tmp_path, [("twothirds", "twothirds")]))
    assert rows[0].met == 2 / 3 and rows[0].reliable == 0.0


# ------------------------------------------------------------------------- ranks


def test_two_identical_entries_share_their_rank_and_neither_ranks_first_alone(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "a", ds, good=2 / 3)
    _run(tmp_path, "b", ds, good=2 / 3)
    _run(tmp_path, "worse", ds, good=0.0)
    rows, _, _ = _build(
        tmp_path, _board(tmp_path, [("a", "a"), ("b", "b"), ("worse", "worse")])
    )
    leaderboard.add_rank_ranges(rows)
    by = {r.label: r for r in rows}
    assert by["a"].rank_range == by["b"].rank_range == (1, 1)
    assert by["a"].first == by["b"].first == 1.0
    assert by["worse"].rank_range == (3, 3)


def test_a_small_difference_gives_a_range_of_ranks_and_not_a_rank(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "a", ds, good=1.0)
    # one ticket in six loses one trial: a 1/18 difference in the met rate
    b = _run(tmp_path, "b", ds, good=1.0)
    lines = (b / "traces.jsonl").read_text(encoding="utf8").splitlines()
    first = json.loads(lines[0])
    first["actions_taken"] = []
    lines[0] = json.dumps(first)
    (b / "traces.jsonl").write_text("\n".join(lines) + "\n", encoding="utf8")
    rows, _, _ = _build(tmp_path, _board(tmp_path, [("a", "a"), ("b", "b")]))
    leaderboard.add_rank_ranges(rows)
    by = {r.label: r for r in rows}
    assert by["a"].rank_range == (1, 1)
    assert by["b"].rank_range == (1, 2)
    assert 0 < by["b"].first < 1


def test_ranking_is_the_same_every_time(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "a", ds, good=2 / 3)
    _run(tmp_path, "b", ds, good=1 / 3)
    board = _board(tmp_path, [("a", "a"), ("b", "b")])
    rows, _, _ = _build(tmp_path, board)
    assert leaderboard.render_board(rows) == leaderboard.render_board(rows)


def test_the_board_can_be_filtered_and_sorted_by_cost(tmp_path):
    ds = _dataset(tmp_path)
    _run(tmp_path, "dear", ds, good=1.0, model="gemini-3.6-flash")
    _run(tmp_path, "cheap", ds, good=2 / 3, model="gemini-3.5-flash-lite")
    board = _board(tmp_path, [("dear", "dear"), ("cheap", "cheap")])
    rows, _, _ = _build(tmp_path, board)
    by_cost = leaderboard.render_board(rows, sort="cost").splitlines()
    assert by_cost[1].split()[1] == "cheap"
    only_good = leaderboard.render_board(rows, min_met=90)
    assert "dear" in only_good and "cheap" not in only_good.split("\n\n")[0]


# ------------------------------------------------------------------- one run each


def test_one_pass_boards_are_all_counted_and_a_flip_shows_up(tmp_path):
    ds = _dataset(tmp_path)
    # both entries meet on a third of the tickets per pass, on different passes
    a = _run(tmp_path, "a", ds, good=1.0)
    b = _run(tmp_path, "b", ds, good=1.0)
    for run, worse_trial in ((a, 1), (b, 2)):
        lines = (run / "traces.jsonl").read_text(encoding="utf8").splitlines()
        rewritten = []
        for line in lines:
            t = json.loads(line)
            if t["trial"] == worse_trial:
                t["actions_taken"] = []
            rewritten.append(json.dumps(t))
        (run / "traces.jsonl").write_text("\n".join(rewritten) + "\n", encoding="utf8")
    board = _board(tmp_path, [("a", "a"), ("b", "b")])
    rows, _, _ = _build(tmp_path, board)
    text = leaderboard.render_first_look(rows, ["a", "b"])
    assert "9 boards are possible" in text
    assert "3 different orders" in text


# ------------------------------------------------------------- the recorded board


def _cli(*args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        assert cli.main(["leaderboard", *args]) == 0
    text = buf.getvalue()
    assert all(len(line) <= 78 for line in text.splitlines())
    return text


def test_the_recorded_board_admits_eight_runs_and_refuses_eight_for_stated_reasons():
    text = _cli("--part", "admit")
    assert text.splitlines()[-1] == "8 admitted, 8 refused"
    assert "triage-live-baseline          refused   no manifest" in text
    assert "41 runs began with leftover state" in text
    assert "harness had uncommitted changes" in text
    assert "another dataset (6311b6, not 27c664)" in text


def test_the_recorded_board_ranks_five_and_blocks_three():
    text = _cli("--part", "board")
    ranked, blocked = text.split("BLOCKED")
    top = ranked.splitlines()[1].split()
    assert top[1:4] == ["3.6", "+", "stall"] and "1-3" in top[0]
    assert len([ln for ln in ranked.splitlines()[1:] if ln[:1].isdigit()]) == 5
    for label in ("2.5-flash", "3.5-lite", "3.5-lite + guard"):
        assert label in blocked


def test_the_first_look_shows_the_order_changing_when_the_run_is_repeated():
    text = _cli("--part", "first-look")
    assert "729 boards are possible" in text
    assert "36 different orders" in text
    assert "3.6 + stall guard                100%" in text


def test_the_html_report_is_one_static_page_with_the_blocked_entries(tmp_path):
    out = tmp_path / "board.html"
    assert cli.main(["leaderboard", "--part", "html", "--out", str(out)]) == 0
    page = out.read_text(encoding="utf8")
    assert page.startswith("<!doctype html>") and "<script" not in page
    assert "Blocked, not ranked" in page and "Refused" in page
    assert "score" not in page.lower().split("<table>")[1].split("</table>")[0]


def test_the_versions_table_shows_a_label_change_moving_the_scores():
    text = _cli("--part", "versions")
    assert "2 of 42 tickets have a different expected result" in text


def test_the_same_two_configurations_have_a_different_gap_on_a_different_set():
    lines = _cli("--part", "sets").splitlines()
    assert lines[3].split()[-2:] == ["+11", "+41"]
