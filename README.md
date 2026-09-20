# agent-evals

A small harness for evaluating AI agents. It grows chapter by chapter with my book *Evaluating AI Agents*, the fourth in the Production AI Agent Engineering series. This first slice is the smallest thing that shows the book's opening claim end to end, and every later chapter extends it only when an experiment needs more.

## The claim it demonstrates

In *Production AI Products* I tested the support-triage system against six tickets and got 6 out of 6. That result is real, and it is also weak evidence. This repository takes the same six tickets and asks what they can and cannot show.

Run against the live model (Gemini, the same `triage-app` code as Book 3, one trial per ticket):

| Measure | Observed | 95% interval |
|---|---|---|
| Routing correct | 6/6 (100%) | 61.0% to 100% |
| Required actions taken | 4/6 (66.7%) | 30.0% to 90.3% |
| Forbidden actions taken | 0/6 | upper bound 39.0% of runs |
| Latency median / p95 | 6.24 s / 8.07 s | |

Three things follow, and each is pinned by a test in `tests/`.

1. **Six cases cannot support "reliable".** A perfect 6/6 is compatible with a true success rate as low as 61%. A gate that asks for the lower end of the interval to reach 80% cannot be passed by any run of fewer than 16 cases (`policies/interval_aware.yaml`).
2. **Book 3's own gate is looser than it looks.** It required a routing pass rate of at least 0.8. With six tickets, 5/6 (83.3%) passes it, so one misrouted ticket goes unnoticed (`policies/book3_original.yaml`).
3. **A quality score cannot see a broken safety boundary.** I gave the technical specialist the refund tool, the kind of change someone makes to be helpful, and replayed the injected tickets with a scripted model that follows the injection. Routing stays 6/6. Two of the six tickets take a forbidden action, and only the hard gate (`invariant_violations <= 0`) blocks the change. The same scripted model against the correct topology takes no forbidden action: the two injected tickets fail closed with an error instead.

There is deliberately no combined score. The scorecard is a profile: each measure with its own uncertainty, and the invariant as a count that must be zero.

## Chapter 2: routing is not success

Scoring the same recording on the work each ticket needs (`required_actions` in the dataset) gives 4 of 6, not 6 of 6. Both billing tickets ended with the specialist asking the customer for an ID the ticket already carried, because `triage-app` builds the model's question from the subject and body only. The adapter `triage-live-customer-id` passes the ID on and scores 6 of 6 on the same measure (`runs/triage-live-customer-id/`, against `runs/triage-live-shipped/` for the product as shipped). Six tickets cannot separate 4/6 from 6/6 by statistics; the reason to believe the fix is the mechanism, not the count. The same change is on the main branch of `triage-app`; this repository still pins the `ch35-end` tag.

```bash
uv run agent-evals stats --run runs/triage-live-shipped --dataset datasets/triage_book3_six.jsonl
uv run agent-evals stats --run runs/triage-live-customer-id --dataset datasets/triage_book3_six.jsonl
```

## Chapter 3: the harness checks itself

Runs can overlap (`--concurrency N`), and every run writes a `manifest.json` with the dataset hash, the model, the exact commit of the product and of the harness, the settings and the wall time. Overlap exposed two ways the harness would have given wrong answers, both reproduced by `scripts/concurrency_probes.py` and pinned by `tests/test_concurrency.py`:

- The product records actions in a context-local list. If the caller touched it before the runs started, every run shared one list. Each run now binds its own.
- An adapter that patched the product for each run left it patched when runs overlapped (the technical specialist ended with seven tools instead of two). Adapters now patch once per batch in `__enter__` and undo it in `__exit__`.

Recorded live runs: `runs/triage-live-3x-sequential` and `runs/triage-live-3x-concurrent`, 18 runs each (6 tickets, 3 trials): 107.72 s against 23.77 s, the same scores.

## Chapter 4: a test set the product has not seen

The six tickets are the product's development examples (the `app crash` runbook entry arrived in the same commit as the first seed ticket), so they are marked `split: dev`. `datasets/triage_heldout_v1.jsonl` is 42 tickets I wrote afterwards, frozen before the first run. `agent-evals dataset check` reports structure, slice sizes and leakage (word overlap, and embedding similarity calibrated with known paraphrases in `datasets/leakage_controls.jsonl`); `agent-evals stats --by KEY` prints a table per slice. See `datasets/triage_heldout.card.md` and `datasets/CHANGELOG.md`.

The first run on the test set (`runs/triage-heldout-v1`, with the chapter 2 fix): routing 30/42 against 6/6 on the development six. Billing 13/13 and security 12/12 route correctly; the technical specialist routes 5/17, because its runbook search only matches the words `app crash` and `login` and the specialist then loops until the tool-round limit. Version 2 of the set (`triage_heldout_v2.jsonl`) corrects two of my labels and marks two as contested; the recorded run can be rescored against it, and the harness says so when the dataset has changed since the run.

## Chapter 5: graders are software

`src/agent_evals/answer_graders.py` has a grader for what an answer says, `asks_for_known_info`, in five versions (v1 to v5), and a grader for a tool argument, `acted_on_the_right_customer`. `agent-evals grader check` measures a grader against hand labels (`datasets/graders/asks_for_known_info.labels.jsonl`: 117 dev answers and 59 test answers, one labeler) and prints precision and recall with intervals; `agent-evals grade` applies a grader to a recorded run. The test answers come from two runs recorded after the versions were frozen and were labeled before any grader ran on them.

Version 4 makes 2 errors in 117 dev answers and 2 in 59 test answers. On the product as shipped it flags about half of the answers as asking for a customer ID the ticket already carried, and 0 with the Chapter 2 fix. The fix also changed what the product does: across the same 42 tickets the shipped product only searches the runbook and escalates, while with the customer ID it looks up 13 invoices, issues 4 refunds and freezes 9 accounts (`runs/triage-heldout-v1-shipped*` against `runs/triage-heldout-v1*`).

## Run it

```bash
uv sync
uv run pytest -q                      # no API key needed

# Reproduce the numbers from the recorded live run
uv run agent-evals stats --run runs/triage-live-baseline --dataset datasets/triage_book3_six.jsonl
uv run agent-evals gate  --run runs/triage-live-baseline --dataset datasets/triage_book3_six.jsonl --policy policies/interval_aware.yaml   # exits 1: too few cases

# Watch the gate catch the regression (scripted model, no key)
uv run agent-evals run  --adapter triage-regressed --dataset datasets/triage_book3_six.jsonl --out runs/triage-regressed
uv run agent-evals gate --run runs/triage-regressed --dataset datasets/triage_book3_six.jsonl --policy policies/book3_original.yaml  # exits 1

# Run the real model yourself (needs GEMINI_API_KEY, loaded at runtime, never committed)
uv run agent-evals run --adapter triage-live --dataset datasets/triage_book3_six.jsonl --out runs/mine --env-file path/to/.env
```

## What is here

- `src/agent_evals/`: the schema (case, trace, scorecard), Wilson intervals, deterministic graders, the runner, gates, and a small CLI.
- `src/agent_evals/adapters/triage.py`: how the harness drives `triage-app` (live, replayed from a recorded run, scripted, and the deliberate regression). The `triage-app` and `reliable-agents-labs` dependencies are pinned to the code the book uses.
- `datasets/triage_book3_six.jsonl`: the six tickets from Book 3, unchanged, with expected specialists and forbidden actions as explicit fields.
- `runs/triage-live-baseline/`: the recorded live run, so anyone can reproduce the scorecard without a key.

## What this does not do yet

One trial per ticket, one product, deterministic graders only. There are no judges, no repeated-trial reliability, no retrieval or multi-agent evaluation, and no online monitoring. Those arrive in later chapters. The recorded run is a single observation of a stochastic system: rerunning it can give a different trace.

## License

MIT, see `LICENSE`.
