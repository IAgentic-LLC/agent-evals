"""Command line: run an adapter over a dataset, print a scorecard, apply a gate."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from agent_evals import answer_graders, grader_check, invariants, tool_calls, world
from agent_evals import dataset as dataset_mod
from agent_evals import gate as gate_mod
from agent_evals.manifest import build_manifest, harness_state, now
from agent_evals.runner import (
    dump_json,
    load_cases,
    read_traces,
    run_cases,
    write_traces,
)
from agent_evals.scorecard import build_scorecard, render_by_slice, render_markdown
from agent_evals.stats import wilson_interval

ADAPTERS = (
    "triage-live",
    "triage-live-customer-id",
    "triage-live-customer-id-topics",
    "triage-live-customer-id-leaky",
    "triage-replay",
    "triage-scripted-wall",
    "triage-regressed",
)


def _make_adapter(name: str, replay: str | None):
    from agent_evals.adapters import triage

    if name == "triage-live":
        return triage.LiveAdapter()
    if name == "triage-live-customer-id":
        return triage.CustomerIdAdapter()
    if name == "triage-live-customer-id-topics":
        return triage.RunbookTopicsAdapter()
    if name == "triage-live-customer-id-leaky":
        return triage.LeakyCustomerIdAdapter()
    if name == "triage-scripted-wall":
        return triage.ScriptedWallAdapter()
    if name == "triage-regressed":
        return triage.RegressedAdapter()
    if name == "triage-replay":
        if not replay:
            raise SystemExit("triage-replay needs --replay PATH to a traces.jsonl")
        return triage.ReplayAdapter(replay)
    raise SystemExit(f"unknown adapter {name!r}")


def _scorecard_for(run_dir: Path, dataset: str, cases_path: str):
    cases = load_cases(cases_path)
    traces = read_traces(run_dir / "traces.jsonl")
    return build_scorecard(dataset, run_dir.name, cases, traces)


def cmd_run(args) -> int:
    if args.env_file:
        load_dotenv(args.env_file)
    if args.adapter.startswith("triage-live") and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            f"{args.adapter} needs GEMINI_API_KEY (use --env-file or export it)"
        )
    cases = load_cases(args.dataset)
    adapter = _make_adapter(args.adapter, args.replay)
    harness, started_at, clock = harness_state(), now(), time.perf_counter()
    traces = asyncio.run(
        run_cases(cases, adapter, trials=args.trials, concurrency=args.concurrency)
    )
    wall_seconds = time.perf_counter() - clock
    out = Path(args.out)
    write_traces(out / "traces.jsonl", traces)
    manifest = build_manifest(
        harness=harness,
        adapter=args.adapter,
        dataset=args.dataset,
        cases=len(cases),
        trials=args.trials,
        concurrency=args.concurrency,
        started_at=started_at,
        wall_seconds=wall_seconds,
    )
    dump_json(out / "manifest.json", manifest)
    sc = build_scorecard(Path(args.dataset).stem, out.name, cases, traces)
    dump_json(out / "scorecard.json", sc.model_dump())
    (out / "scorecard.md").write_text(render_markdown(sc), encoding="utf8")
    print(render_markdown(sc))
    return 0


def _dataset_note(run_dir: Path, dataset: str) -> str:
    """Say so when a run is scored against a different dataset than it ran on."""
    try:
        recorded = json.loads((run_dir / "manifest.json").read_text(encoding="utf8"))
        ran_on = recorded["dataset"]["sha256"]
    except (OSError, KeyError, ValueError):
        return ""
    now_hash = hashlib.sha256(Path(dataset).read_bytes()).hexdigest()
    if ran_on == now_hash:
        return ""
    return (
        "Note: this run was recorded against a different version of the dataset\n"
        f"(hash {ran_on[:8]}, now {now_hash[:8]}). Scoring it against the current\n"
        "file is allowed, and it is not the same test.\n\n"
    )


def cmd_stats(args) -> int:
    run_dir = Path(args.run)
    sc = _scorecard_for(run_dir, Path(args.dataset).stem, args.dataset)
    print(_dataset_note(run_dir, args.dataset) + render_markdown(sc), end="")
    if args.by:
        cases = load_cases(args.dataset)
        traces = read_traces(run_dir / "traces.jsonl")
        if args.derived:
            derived = json.loads(Path(args.derived).read_text(encoding="utf8"))
            for case in cases:
                case.slices[derived["key"]] = derived["values"].get(
                    case.case_id, "(none)"
                )
        for key in args.by:
            print()
            print(render_by_slice(cases, traces, key), end="")
    return 0


def cmd_dataset_check(args) -> int:
    if args.env_file:
        load_dotenv(args.env_file)
    cases = load_cases(args.dataset)
    issues = dataset_mod.check_cases(cases)
    reference = load_cases(args.against) if args.against else None
    lexical = dataset_mod.nearest_lexical(cases, reference) if reference else None
    semantic = controls = None
    if args.embeddings:
        if reference is None:
            raise SystemExit("--embeddings needs --against")
        if not os.environ.get("GEMINI_API_KEY"):
            raise SystemExit("--embeddings needs GEMINI_API_KEY (use --env-file)")
        from reliable_agents_labs.models import build_embedding_client

        embed = build_embedding_client().embed
        pairs = []
        if args.controls:
            for line in Path(args.controls).read_text(encoding="utf8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    pairs.append((row["source"], row["text"]))

        async def scores():
            near = await dataset_mod.nearest_semantic(cases, reference, embed)
            ctl = await dataset_mod.control_scores(pairs, reference, embed)
            return near, ctl

        semantic, controls = asyncio.run(scores())
        if args.write_derived:
            import yaml

            config = yaml.safe_load(
                Path("config/models.yaml").read_text(encoding="utf8")
            )
            flags = dataset_mod.near_dev_flags(
                semantic, controls, config["embedding_model"]["model_id"]
            )
            dump_json(args.write_derived, flags)
    print(
        dataset_mod.render_check(
            Path(args.dataset).name, cases, issues, lexical, semantic, controls
        ),
        end="",
    )
    return 1 if any(i.severity == "error" for i in issues) else 0


def _grader(spec: str):
    """`asks_for_known_info:v4` -> the function, or exit with a clear message."""
    name, _, version = spec.partition(":")
    versions = answer_graders.GRADERS.get(name)
    if versions is None or version not in versions:
        known = ", ".join(
            f"{n}:{v}" for n, vs in answer_graders.GRADERS.items() for v in vs
        )
        raise SystemExit(f"unknown grader {spec!r}; choose one of {known}")
    return versions[version]


def cmd_grader_check(args) -> int:
    labels = grader_check.load_labels(args.labels)
    if args.split:
        labels = [row for row in labels if row["split"] == args.split]
    cases = grader_check.load_cases_by_id(*args.datasets)
    result = grader_check.check(_grader(args.grader), labels, args.runs_dir, cases)
    name = args.grader + (f" on the {args.split} split" if args.split else "")
    print(grader_check.render(name, result), end="")
    return 0


def cmd_grade(args) -> int:
    cases = {c.case_id: c for c in load_cases(args.dataset)}
    grader = _grader(args.grader)
    traces = [
        t
        for t in read_traces(Path(args.run) / "traces.jsonl")
        if not t.error and t.answer.strip()
    ]
    flagged = sum(grader(cases[t.case_id], t) for t in traces)
    low, high = wilson_interval(flagged, len(traces))
    print(
        f"{args.grader}: flagged {flagged} of {len(traces)} answers "
        f"({100 * flagged / len(traces):.1f}%, 95% interval "
        f"{100 * low:.1f}% to {100 * high:.1f}%)"
    )
    return 0


def cmd_state(args) -> int:
    from triage_app.tools import _INVOICES

    cases = {c.case_id: c for c in load_cases(args.dataset)}
    traces = read_traces(Path(args.run) / "traces.jsonl")
    found: dict[str, list[str]] = {rule: [] for rule in world.RULES}
    for t in traces:
        for rule in world.state_violations(cases[t.case_id], t, _INVOICES):
            found[rule].append(f"{t.case_id} t{t.trial}")
    print(f"State rules for {args.run} ({len(traces)} traces)")
    for rule, hits in found.items():
        shown = ", ".join(hits[:2])
        more = f" (+{len(hits) - 2} more)" if len(hits) > 2 else ""
        print(f"  {rule:<30}{len(hits):>3}   {shown}{more}".rstrip())
    return 1 if any(found.values()) else 0


def _invariant_findings(
    run: str, dataset: str, policy: str, permissions_path: str | None
) -> tuple[dict[str, list[str]], set[str], int]:
    """Rule hits by rule, the traces that broke any rule, and the trace count."""
    cases = {c.case_id: c for c in load_cases(dataset)}
    traces = read_traces(Path(run) / "traces.jsonl")
    permissions = (
        invariants.load_permissions(permissions_path) if permissions_path else None
    )
    found: dict[str, list[str]] = {rule: [] for rule in invariants.RULES}
    broken: set[str] = set()
    for t in traces:
        for rule in invariants.violations(cases[t.case_id], t, policy, permissions):
            found[rule].append(f"{t.case_id} t{t.trial}")
            broken.add(f"{t.case_id} t{t.trial}")
    return found, broken, len(traces)


def cmd_invariants(args) -> int:
    found, broken, total = _invariant_findings(
        args.run, args.dataset, args.policy, args.permissions
    )
    print(f"Protected invariants for {args.run} ({total} traces)")
    print(f"policy {args.policy}: {invariants.POLICIES[args.policy]}")
    # A deny-list has one rule. The allow-list rows would always read 0.
    for rule in invariants.RULES[: 1 if args.policy == "deny-list" else None]:
        hits = found[rule]
        shown = ", ".join(hits[:2])
        more = f" (+{len(hits) - 2} more)" if len(hits) > 2 else ""
        print(f"  {rule:<30}{len(hits):>3}   {shown}{more}".rstrip())
    print(f"traces that broke a rule: {len(broken)} of {total}")
    return 1 if broken else 0


def cmd_calls(args) -> int:
    cases = {c.case_id: c for c in load_cases(args.dataset)}
    traces = read_traces(Path(args.run) / "traces.jsonl")
    if not any(t.tool_calls for t in traces):
        print(
            f"{args.run} has no recorded tool calls. It was recorded before they were."
        )
        return 2
    print(tool_calls.render(args.run, cases, traces), end="")
    broken = any(tool_calls.call_problems(cases[t.case_id], t) for t in traces)
    return 1 if broken else 0


def cmd_gate(args) -> int:
    sc = _scorecard_for(Path(args.run), Path(args.dataset).stem, args.dataset)
    extra = None
    if args.invariants:
        _, broken, _ = _invariant_findings(
            args.run, args.dataset, args.invariants, args.permissions
        )
        extra = {"protected_invariant_violations": float(len(broken))}
    result = gate_mod.evaluate(gate_mod.load_policy(args.policy), sc, extra)
    print(gate_mod.render(result))
    return 0 if result.passed else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-evals")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser(
        "run", help="run an adapter over a dataset and write a run directory"
    )
    p_run.add_argument("--adapter", choices=ADAPTERS, required=True)
    p_run.add_argument("--dataset", required=True)
    p_run.add_argument("--out", required=True)
    p_run.add_argument("--trials", type=int, default=1)
    p_run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="how many runs may be in flight at once (default 1)",
    )
    p_run.add_argument("--replay", help="traces.jsonl to replay (triage-replay only)")
    p_run.add_argument(
        "--env-file", help="a .env file to load at runtime (never committed)"
    )
    p_run.set_defaults(func=cmd_run)

    p_stats = sub.add_parser("stats", help="print the scorecard for a run directory")
    p_stats.add_argument("--run", required=True)
    p_stats.add_argument("--dataset", required=True)
    p_stats.add_argument(
        "--by", action="append", help="also print a table per value of this slice key"
    )
    p_stats.add_argument(
        "--derived", help="a derived slice written by `dataset check --write-derived`"
    )
    p_stats.set_defaults(func=cmd_stats)

    p_dataset = sub.add_parser("dataset", help="check an evaluation dataset")
    ds_sub = p_dataset.add_subparsers(dest="dataset_command", required=True)
    p_check = ds_sub.add_parser("check", help="structure, slices and leakage")
    p_check.add_argument("--dataset", required=True)
    p_check.add_argument("--against", help="reference set the cases must not resemble")
    p_check.add_argument("--controls", help="known paraphrases, as a yardstick")
    p_check.add_argument(
        "--embeddings", action="store_true", help="add the semantic check"
    )
    p_check.add_argument(
        "--write-derived", help="write the near_dev slice to this JSON file"
    )
    p_check.add_argument("--env-file", help="a .env file to load at runtime")
    p_check.set_defaults(func=cmd_dataset_check)

    p_grader = sub.add_parser("grader", help="test a grader against hand labels")
    gr_sub = p_grader.add_subparsers(dest="grader_command", required=True)
    p_gcheck = gr_sub.add_parser("check", help="precision and recall against labels")
    p_gcheck.add_argument("--grader", required=True, help="for example name:v4")
    p_gcheck.add_argument("--labels", required=True)
    p_gcheck.add_argument("--split", help="use only the dev or the test labels")
    p_gcheck.add_argument("--runs-dir", default="runs")
    p_gcheck.add_argument(
        "--datasets",
        nargs="+",
        default=[
            "datasets/triage_book3_six.jsonl",
            "datasets/triage_heldout_v1.jsonl",
        ],
    )
    p_gcheck.set_defaults(func=cmd_grader_check)

    p_grade = sub.add_parser("grade", help="apply a grader to a run's answers")
    p_grade.add_argument("--run", required=True)
    p_grade.add_argument("--dataset", required=True)
    p_grade.add_argument("--grader", required=True, help="for example name:v4")
    p_grade.set_defaults(func=cmd_grade)

    p_state = sub.add_parser("state", help="check rules about the world after each run")
    p_state.add_argument("--run", required=True)
    p_state.add_argument("--dataset", required=True)
    p_state.set_defaults(func=cmd_state)

    p_inv = sub.add_parser(
        "invariants", help="check protected invariants after each run; exit 1 if broken"
    )
    p_inv.add_argument("--run", required=True)
    p_inv.add_argument("--dataset", required=True)
    p_inv.add_argument("--policy", choices=list(invariants.POLICIES), required=True)
    p_inv.add_argument("--permissions", help="what each ticket itself asked for")
    p_inv.set_defaults(func=cmd_invariants)

    p_calls = sub.add_parser(
        "calls", help="check every recorded tool call; exit 1 if a rule was broken"
    )
    p_calls.add_argument("--run", required=True)
    p_calls.add_argument("--dataset", required=True)
    p_calls.set_defaults(func=cmd_calls)

    p_gate = sub.add_parser("gate", help="apply a gate policy; exit 1 if blocked")
    p_gate.add_argument("--run", required=True)
    p_gate.add_argument("--dataset", required=True)
    p_gate.add_argument("--policy", required=True)
    p_gate.add_argument(
        "--invariants",
        choices=list(invariants.POLICIES),
        help="also count traces that break this invariant policy",
    )
    p_gate.add_argument("--permissions", help="what each ticket itself asked for")
    p_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
