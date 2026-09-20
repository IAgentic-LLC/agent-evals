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

from agent_evals import (
    abstention,
    answer_graders,
    compare,
    conversation,
    forced,
    grader_check,
    invariants,
    judge,
    judge_report,
    labels,
    tool_calls,
    trajectory,
    world,
)
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
    "triage-live-customer-id-stall-guard",
    "triage-live-customer-id-leaky",
    "reorder-live",
    "reorder-scripted",
    "pkg-live",
    "pkg-live-permissive",
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
    if name == "triage-live-customer-id-stall-guard":
        return triage.StallGuardAdapter()
    if name == "triage-live-customer-id-topics":
        return triage.RunbookTopicsAdapter()
    if name == "reorder-live":
        from agent_evals.adapters import reorder

        return reorder.ReorderAdapter()
    if name == "reorder-scripted":
        from agent_evals.adapters import reorder

        return reorder.ReorderScriptedAdapter()
    if name == "pkg-live":
        from agent_evals.adapters import pkgintel

        return pkgintel.PkgAnswerAdapter()
    if name == "pkg-live-permissive":
        from agent_evals.adapters import pkgintel

        return pkgintel.PermissivePromptAdapter()
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
    live = args.adapter.startswith(("triage-live", "reorder-live", "pkg-live"))
    if live and not os.environ.get("GEMINI_API_KEY"):
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
    if args.adapter.startswith("pkg"):
        print(_grounding_text(out.name, cases, traces), end="")
        return 0
    if args.adapter.startswith("reorder"):
        # A conversation is not scored as routing, so it has no scorecard.
        print(
            conversation.render(out.name, {c.case_id: c for c in cases}, traces), end=""
        )
        return 0
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


def cmd_trajectory(args) -> int:
    cases = {c.case_id: c for c in load_cases(args.dataset)}
    traces = read_traces(Path(args.run) / "traces.jsonl")
    if not any(t.tool_calls for t in traces):
        print(
            f"{args.run} has no recorded tool calls. It was recorded before they were."
        )
        return 2
    found: dict[str, list[str]] = {rule: [] for rule in trajectory.RULES}
    broken: set[str] = set()
    for t in traces:
        label = f"{t.case_id} t{t.trial}"
        for rule in trajectory.violations(
            cases[t.case_id], t, args.max_calls, args.max_stall
        ):
            found[rule].append(label)
            broken.add(label)
    print(f"Trajectory constraints for {args.run} ({len(traces)} traces)")
    print(
        f"limits: {args.max_calls} tool calls, {args.max_stall} empty or irrelevant in a row"
    )
    for rule, hits in found.items():
        shown = ", ".join(hits[:2])
        more = f" (+{len(hits) - 2} more)" if len(hits) > 2 else ""
        print(f"  {rule:<22}{len(hits):>3}   {shown}{more}".rstrip())
    print(f"traces that broke a constraint: {len(broken)} of {len(traces)}")
    return 1 if broken else 0


def _grounding_names() -> list[str]:
    from agent_evals import retrieval

    return [r["name"] for r in retrieval.load_corpus(GROUNDING_CORPUS)]


GROUNDING_CORPUS = Path(__file__).resolve().parents[2] / "datasets/pkg_corpus_v1.jsonl"


def _grounding_text(label, cases, traces) -> str:
    from agent_evals import grounding

    by_id = {c.case_id: c for c in cases}
    return grounding.render_summary(label, by_id, traces, _grounding_names())


def cmd_grounding_grade(args) -> int:
    from agent_evals import grounding

    cases = {c.case_id: c for c in load_cases(args.dataset)}
    traces = [t for run in args.run for t in read_traces(Path(run) / "traces.jsonl")]
    label = " + ".join(Path(run).name for run in args.run)
    names = _grounding_names()
    parts = {
        "summary": lambda: grounding.render_summary(label, cases, traces, names),
        "bounds": lambda: grounding.render_bounds(cases, traces, names),
        "fresh": lambda: grounding.render_fresh(cases, traces),
        "counterfactual": lambda: grounding.render_counterfactual(cases, traces),
    }
    if args.part in ("readings", "refusals"):
        runs = {Path(r).name: read_traces(Path(r) / "traces.jsonl") for r in args.run}
        if args.part == "refusals":
            print(grounding.render_refusals(cases, runs), end="")
        else:
            readings = grounding.load_readings(args.readings)
            print(grounding.render_readings(cases, runs, readings, names), end="")
    elif args.part == "flagged":
        for trace, hits in grounding.flagged(cases, traces, names):
            print(f"{trace.case_id} t{trace.trial}: {', '.join(hits)}")
    else:
        print(parts[args.part](), end="")
    # A citation the answer takes from outside what was retrieved breaks the
    # product's own contract, so it fails the command.
    broken = any(
        grounding.check(cases[t.case_id], t, names)["invalid_citation"] for t in traces
    )
    return 1 if broken else 0


def cmd_grounding_check(args) -> int:
    from agent_evals import grounding_check

    return grounding_check.main(_grounding_names())


def _abstention_inputs(args):
    cases = {c.case_id: c for c in load_cases(args.dataset)}
    runs = {
        name: [read_traces(Path("runs") / r / "traces.jsonl") for r in dirs]
        for name, dirs in abstention.CONFIGS.items()
    }
    return cases, runs


def cmd_abstention_report(args) -> int:
    cases, runs = _abstention_inputs(args)
    first = runs["shipped prompt"][0]
    parts = {
        "matrix": lambda: abstention.render_matrix(cases, runs),
        "kinds": lambda: abstention.render_kinds(cases, runs),
        "paired": lambda: abstention.render_paired(cases, runs),
        "scores": lambda: abstention.render_scores(cases, first),
        "gate": lambda: abstention.render_gate(cases, first),
        "caught": lambda: abstention.render_caught(cases, first),
        "overlap": lambda: abstention.render_overlap(cases, runs),
    }
    print(parts[args.part](), end="")
    return 0


def cmd_judge_run(args) -> int:
    if args.env_file:
        load_dotenv(args.env_file)
    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("judge run needs GEMINI_API_KEY (use --env-file or export it)")

    items = judge.load_items(args.items)
    if args.split:
        items = [i for i in items if i["split"] == args.split]
    client = _judge_client(args.model)
    harness, started, clock = harness_state(), now(), time.perf_counter()
    rows = asyncio.run(
        judge.judge_all(
            items, client, args.version, args.passes, concurrency=args.concurrency
        )
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "judgments.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows),
        encoding="utf8",
    )
    dump_json(
        out / "manifest.json",
        {
            "judge_model": _judge_model_config(args.model),
            "prompt": {
                "version": args.version,
                "sha256": judge.prompt_hash(args.version),
                "text": judge.VERSIONS[args.version],
            },
            "items": {
                "path": args.items,
                "sha256": hashlib.sha256(Path(args.items).read_bytes()).hexdigest(),
                "count": len(items),
                "split": args.split,
            },
            "passes": args.passes,
            "concurrency": args.concurrency,
            "started_at": started.isoformat(timespec="seconds"),
            "wall_seconds": round(time.perf_counter() - clock, 2),
            "harness": harness,
        },
    )
    print(judge_report.render_cost(rows), end="")
    return 0


def _judge_client(model: str | None):
    from reliable_agents_labs.models import (
        GeminiOpenAICompatibleClient,
        build_model_client,
    )

    if model:
        return GeminiOpenAICompatibleClient(model_id=model)
    return build_model_client("judge_model")


def _judge_model_config(model: str | None = None):
    import yaml

    if model:
        return {"provider": "gemini", "model_id": model, "override": True}

    text = Path("config/models.yaml").read_text(encoding="utf8")
    role = yaml.safe_load(text)["judge_model"]
    return {"provider": role.get("provider"), "model_id": role.get("model_id")}


def cmd_judge_perturb(args) -> int:
    from agent_evals import bias

    kinds = bias.ATTACKS if args.set == "attacks" else bias.PERTURBATIONS
    changed = bias.perturb_all(judge.load_items(args.items), kinds)
    Path(args.out).write_text(
        "".join(
            json.dumps(i, ensure_ascii=False, separators=(",", ":")) + "\n"
            for i in changed
        ),
        encoding="utf8",
        newline="\n",
    )
    print(f"wrote {len(changed)} changed items to {args.out}")
    return 0


def cmd_judge_report(args) -> int:
    items = judge.load_items(args.items)
    rows = judge_report.load_rows(args.run)
    split = args.split
    if args.part == "flips":
        base = judge_report.load_rows([args.run[0]])
        moved = judge_report.load_rows([args.perturbed_run])
        changed = judge.load_items(args.perturbed_items)
        print(judge_report.render_flips(items, base, changed, moved, split), end="")
        return 0
    if args.part == "agreement":
        named = {
            Path(r).name.replace("judge-", ""): judge_report.load_rows([r])
            for r in args.run
        }
        print(judge_report.render_agreement(items, named, split), end="")
        return 0
    if args.part == "judges":
        named = {
            Path(r).name.replace("judge-", ""): judge_report.load_rows([r])
            for r in args.run
        }
        print(judge_report.render_judges(items, named, split), end="")
        return 0
    if args.part == "compare":
        first, second = (judge_report.load_rows([r]) for r in args.run)
        names = [Path(r).name for r in args.run]
        print(judge_report.render_compare(items, first, second, names, split), end="")
        return 0
    if args.part == "disagreements":
        for item, claim in judge_report.disagreements(items, rows, split):
            print(f"{item['item_id']}: {claim}")
        return 0
    parts = {
        "planted": lambda: judge_report.render_planted(items, rows, split),
        "real": lambda: judge_report.render_real(items, rows, split),
        "retest": lambda: judge_report.render_retest(items, rows, split),
        "cost": lambda: judge_report.render_cost(rows, args.price_in, args.price_out),
    }
    print(parts[args.part](), end="")
    return 0


def cmd_compare(args) -> int:
    part = args.part
    if part == "coverage":
        print(compare.render_coverage(), end="")
    elif part == "paired-coverage":
        print(compare.render_paired_coverage(), end="")
    elif part == "attack":
        print(compare.attack_report(), end="")
    else:
        cases, runs = _abstention_inputs(args)
        report = {
            "intervals": compare.render_intervals,
            "difference": compare.render_difference,
            "noise": compare.render_noise,
            "slices": compare.render_slices,
            "plan": compare.render_plan,
        }[part]
        print(report(cases, runs), end="")
    return 0


def cmd_labels(args) -> int:
    items = labels.item_list()
    truth = labels.author_labels(items, args.convention, args.adjudicated)
    part = args.part
    if part == "matrix":
        raters = {
            Path(r).name.replace("judge-", ""): labels.judge_labels(r)
            for r in args.judge
        }
        print(labels.render_matrix(items, raters, truth, args.group), end="")
    elif part == "retest":
        print(labels.render_retest(items, [Path(r).name for r in args.judge]), end="")
    elif part == "sensitivity":
        print(
            labels.render_sensitivity(items, truth, labels.judge_labels(args.judge[0])),
            end="",
        )
    elif part == "calibrate":
        name = Path(args.judge[0]).name.replace("judge-", "")
        print(
            labels.render_calibration(
                items, truth, labels.judge_labels(args.judge[0]), name
            ),
            end="",
        )
    elif part == "plan":
        print(
            labels.render_plan(0.85, 0.97, 0.08, (50, 100, 200, 400, 800), (100, 400)),
            end="",
        )
    elif part == "second":
        from agent_evals import agreement

        second = agreement.load_labels(args.second)
        judged = labels.judge_labels(args.judge[0])
        print(labels.render_second(items, second, truth, judged), end="")
    return 0


def _forced_inputs(args):
    runs = {
        r: read_traces(Path("runs") / r / "traces.jsonl")
        for runs in forced.GROUPS.values()
        for r in runs
    }
    return forced.forced_answers(runs), forced.load_readings(args.readings)


def cmd_forced_report(args) -> int:
    found, readings = _forced_inputs(args)
    if args.part == "readings":
        print(forced.render_readings(found, readings), end="")
    else:
        print(forced.render_proxies(found, readings), end="")
    return 0


def cmd_conversation_check(args) -> int:
    from agent_evals import mechanics

    # The real workflow must pass every check, and every planted fault must fail one.
    wrong = False
    for variant in mechanics.VARIANTS.values():
        found = asyncio.run(mechanics.run_mechanics(variant))
        failed = {name: why for name, why in found.items() if why}
        is_real = variant.name == "the real workflow"
        wrong = wrong or (bool(failed) if is_real else not failed)
        print(f"{variant.name}: {len(failed)} of {len(found)} checks failed")
        for name, why in failed.items():
            print(f"  {name}: {why}"[:78])
    return 1 if wrong else 0


def cmd_conversation_grade(args) -> int:
    cases = {c.case_id: c for c in load_cases(args.dataset)}
    traces = [t for run in args.run for t in read_traces(Path(run) / "traces.jsonl")]
    names = " + ".join(Path(run).name for run in args.run)
    print(conversation.render(names, cases, traces), end="")
    broken = any(conversation.violated_invariants(cases[t.case_id], t) for t in traces)
    return 1 if broken else 0


def _study(args):
    from agent_evals import retrieval, retrieval_report

    run = Path(args.run)
    study = retrieval_report.Study(
        retrieval.load_corpus(args.corpus),
        load_cases(args.queries),
        retrieval.load_vectors(run / "embeddings-shipped.npz"),
        retrieval.load_vectors(run / "embeddings-typed.npz"),
    )
    return retrieval, retrieval_report, asyncio.run(study.run())


def cmd_retrieval_report(args) -> int:
    _, report, study = _study(args)
    parts = {
        "methods": lambda: report.render_methods(study, Path(args.run).name),
        "kinds": lambda: report.render_kinds(study),
        "paired": lambda: report.render_paired(study),
        "misses": lambda: report.render_misses(study),
        "scores": lambda: report.render_scores(study),
    }
    print(parts[args.part](), end="")
    return 0


def cmd_retrieval_isolation(args) -> int:
    retrieval, _, study = _study(args)
    vectors = retrieval.load_vectors(Path(args.run) / "embeddings-shipped.npz")
    print("Results that belong to another tenant, over every question:")
    wrong = False
    for shared, label in (
        (False, "one collection per tenant"),
        (True, "one shared collection"),
    ):
        leaked, total = asyncio.run(
            retrieval.leaked_results(
                study.corpus, study.queries, vectors, shared=shared
            )
        )
        wrong = wrong or (leaked > 0 if not shared else leaked == 0)
        print(f"  {label:<28}{leaked:>4} of {total}")
    return 1 if wrong else 0


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

    p_traj = sub.add_parser(
        "trajectory", help="check the shape of each run; exit 1 if a constraint broke"
    )
    p_traj.add_argument("--run", required=True)
    p_traj.add_argument("--dataset", required=True)
    p_traj.add_argument(
        "--max-calls", type=int, required=True, help="tool calls a run may make"
    )
    p_traj.add_argument(
        "--max-stall",
        type=int,
        required=True,
        help="empty or irrelevant results in a row that count as a stall",
    )
    p_traj.set_defaults(func=cmd_trajectory)

    p_conv = sub.add_parser(
        "conversation", help="check a workflow that pauses and resumes"
    )
    conv_sub = p_conv.add_subparsers(dest="conversation_command", required=True)
    p_cc = conv_sub.add_parser(
        "check", help="the mechanics, on the real workflow and five faulty ones"
    )
    p_cc.set_defaults(func=cmd_conversation_check)
    p_cg = conv_sub.add_parser("grade", help="grade recorded conversations")
    p_cg.add_argument("--run", nargs="+", required=True)
    p_cg.add_argument("--dataset", required=True)
    p_cg.set_defaults(func=cmd_conversation_grade)

    p_ab = sub.add_parser("abstention", help="score declining, from recorded runs")
    p_ab.add_argument("--dataset", required=True)
    p_ab.add_argument(
        "--part",
        choices=("matrix", "kinds", "paired", "scores", "gate", "caught", "overlap"),
        required=True,
    )
    p_ab.set_defaults(func=cmd_abstention_report)
    p_ju = sub.add_parser("judge", help="run and read an LLM judge")
    ju_sub = p_ju.add_subparsers(dest="judge_command", required=True)
    p_jr = ju_sub.add_parser("run", help="judge every item with the live judge model")
    p_jr.add_argument("--items", required=True)
    p_jr.add_argument("--version", choices=sorted(judge.VERSIONS), required=True)
    p_jr.add_argument("--out", required=True)
    p_jr.add_argument("--passes", type=int, default=2)
    p_jr.add_argument("--concurrency", type=int, default=6)
    p_jr.add_argument("--split", choices=("dev", "test"))
    p_jr.add_argument("--env-file")
    p_jr.add_argument("--model", help="judge with this model id instead of the config")
    p_jr.set_defaults(func=cmd_judge_run)
    p_jb = ju_sub.add_parser("perturb", help="make changed copies of the items")
    p_jb.add_argument("--items", required=True)
    p_jb.add_argument("--out", required=True)
    p_jb.add_argument("--set", choices=("basic", "attacks"), default="basic")
    p_jb.set_defaults(func=cmd_judge_perturb)
    p_jp = ju_sub.add_parser("report", help="read recorded verdicts")
    p_jp.add_argument("--run", nargs="+", required=True)
    p_jp.add_argument("--items", required=True)
    p_jp.add_argument("--split", choices=("dev", "test", "unseen"))
    p_jp.add_argument("--perturbed-items")
    p_jp.add_argument("--perturbed-run")
    p_jp.add_argument("--price-in", type=float, help="dollars per million input tokens")
    p_jp.add_argument(
        "--price-out", type=float, help="dollars per million output tokens"
    )
    p_jp.add_argument(
        "--part",
        choices=(
            "planted",
            "real",
            "retest",
            "cost",
            "disagreements",
            "compare",
            "flips",
            "judges",
            "agreement",
        ),
        required=True,
    )
    p_jp.set_defaults(func=cmd_judge_report)
    p_cp = sub.add_parser("compare", help="is one run really different from another")
    p_cp.add_argument(
        "--part",
        required=True,
        choices=(
            "intervals",
            "difference",
            "noise",
            "slices",
            "plan",
            "coverage",
            "paired-coverage",
            "attack",
        ),
    )
    p_cp.add_argument("--dataset", default="datasets/pkg_abstain_v1.jsonl")
    p_cp.set_defaults(func=cmd_compare)

    p_lb = sub.add_parser("labels", help="agreement between people and judges")
    p_lb.add_argument(
        "--part",
        choices=("matrix", "retest", "sensitivity", "calibrate", "plan", "second"),
        required=True,
    )
    p_lb.add_argument("--judge", nargs="*", default=[])
    p_lb.add_argument("--group", choices=("real", "all", "planted"), default="real")
    p_lb.add_argument("--convention", choices=("strict", "lenient"), default="strict")
    p_lb.add_argument("--adjudicated", action="store_true")
    p_lb.add_argument("--second", help="a label file from a second person")
    p_lb.set_defaults(func=cmd_labels)
    p_fo = sub.add_parser("forced", help="the answers that follow three empty searches")
    p_fo.add_argument("--readings", default="datasets/triage_forced.readings.jsonl")
    p_fo.add_argument("--part", choices=("readings", "proxies"), required=True)
    p_fo.set_defaults(func=cmd_forced_report)

    p_gr = sub.add_parser("grounding", help="check answers against what was retrieved")
    gr_sub = p_gr.add_subparsers(dest="grounding_command", required=True)
    p_gg = gr_sub.add_parser("grade", help="grade recorded answers")
    p_gg.add_argument("--run", nargs="+", required=True)
    p_gg.add_argument("--dataset", required=True)
    p_gg.add_argument(
        "--part",
        choices=(
            "summary",
            "bounds",
            "fresh",
            "counterfactual",
            "flagged",
            "readings",
            "refusals",
        ),
        default="summary",
    )
    p_gg.add_argument("--readings", default="datasets/pkg_answers_v1.readings.jsonl")
    p_gg.set_defaults(func=cmd_grounding_grade)
    p_gc = gr_sub.add_parser(
        "check", help="the checks, on a faithful script and six faulty ones"
    )
    p_gc.set_defaults(func=cmd_grounding_check)

    p_ret = sub.add_parser("retrieval", help="score retrieval from recorded embeddings")
    ret_sub = p_ret.add_subparsers(dest="retrieval_command", required=True)
    for name, helptext in (
        ("report", "tables of hit rate, recall and rank for each method"),
        ("isolation", "count results that belong to another tenant"),
    ):
        p_r = ret_sub.add_parser(name, help=helptext)
        p_r.add_argument(
            "--run", required=True, help="folder with the recorded vectors"
        )
        p_r.add_argument("--queries", required=True)
        p_r.add_argument("--corpus", required=True)
        if name == "report":
            p_r.add_argument(
                "--part",
                choices=("methods", "kinds", "paired", "misses", "scores"),
                required=True,
            )
            p_r.set_defaults(func=cmd_retrieval_report)
        else:
            p_r.set_defaults(func=cmd_retrieval_isolation)

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
