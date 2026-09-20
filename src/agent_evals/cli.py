"""Command line: run an adapter over a dataset, print a scorecard, apply a gate."""

import argparse
import asyncio
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from agent_evals import gate as gate_mod
from agent_evals.runner import (
    dump_json,
    load_cases,
    read_traces,
    run_cases,
    write_traces,
)
from agent_evals.scorecard import build_scorecard, render_markdown

ADAPTERS = ("triage-live", "triage-replay", "triage-scripted-wall", "triage-regressed")


def _make_adapter(name: str, replay: str | None):
    from agent_evals.adapters import triage

    if name == "triage-live":
        return triage.LiveAdapter()
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
    if args.adapter == "triage-live" and not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit(
            "triage-live needs GEMINI_API_KEY (use --env-file or export it)"
        )
    cases = load_cases(args.dataset)
    adapter = _make_adapter(args.adapter, args.replay)
    traces = asyncio.run(run_cases(cases, adapter, trials=args.trials))
    out = Path(args.out)
    write_traces(out / "traces.jsonl", traces)
    sc = build_scorecard(Path(args.dataset).stem, out.name, cases, traces)
    dump_json(out / "scorecard.json", sc.model_dump())
    (out / "scorecard.md").write_text(render_markdown(sc), encoding="utf8")
    print(render_markdown(sc))
    return 0


def cmd_stats(args) -> int:
    sc = _scorecard_for(Path(args.run), Path(args.dataset).stem, args.dataset)
    print(render_markdown(sc))
    return 0


def cmd_gate(args) -> int:
    sc = _scorecard_for(Path(args.run), Path(args.dataset).stem, args.dataset)
    result = gate_mod.evaluate(gate_mod.load_policy(args.policy), sc)
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
    p_run.add_argument("--replay", help="traces.jsonl to replay (triage-replay only)")
    p_run.add_argument(
        "--env-file", help="a .env file to load at runtime (never committed)"
    )
    p_run.set_defaults(func=cmd_run)

    p_stats = sub.add_parser("stats", help="print the scorecard for a run directory")
    p_stats.add_argument("--run", required=True)
    p_stats.add_argument("--dataset", required=True)
    p_stats.set_defaults(func=cmd_stats)

    p_gate = sub.add_parser("gate", help="apply a gate policy; exit 1 if blocked")
    p_gate.add_argument("--run", required=True)
    p_gate.add_argument("--dataset", required=True)
    p_gate.add_argument("--policy", required=True)
    p_gate.set_defaults(func=cmd_gate)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
