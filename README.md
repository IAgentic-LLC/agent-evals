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

## Chapter 6: state beats prose

`src/agent_evals/world.py` rebuilds what a run changed (refunds, frozen accounts, restarts, escalations) from the ledger of side effects, and checks four rules about it (Chapter 7 adds a fifth): a run acts only on the ticket's own customer, a refund never exceeds the invoice on file, at most one refund, and a run that fails leaves the world unchanged. `agent-evals state --run R --dataset D` prints the rule counts. `src/agent_evals/action_claims.py` has three ways to ask whether an action happened (a naive prose check, a careful prose check in English, Spanish and French, and the state), and `scripts/prose_vs_state.py` compares them over the 176 recorded answers.

The prose graders disagree with the state a lot (a naive refund check is right 8 times in 32), the state cannot be misled by wording, and across 48 errored runs one had already restarted a service and left no answer (`runs/triage-heldout-v1`, HO-042).

## Chapter 7: sandboxes and simulated environments

Each run needs its own state. `_run_once` in `src/agent_evals/adapters/triage.py` gives every run a new ledger and records how many actions the ledger already held (`ledger_at_start`, on every trace). A fifth state rule, `started_with_leftover_state`, fires when that number is above zero. `LeakyCustomerIdAdapter` skips the reset on purpose, and `runs/triage-heldout-v1-leaky` is its recorded run on the 42 held-out tickets; it must never be used to measure anything. `runs/triage-heldout-v1-3` is a third clean pass, beside `triage-heldout-v1` and `-2`.

On that run 41 of 42 tickets started with leftovers, the leaky score was 31 required-action passes against 25 to 28 on the clean runs (29 when each ticket's own actions are counted), and 5 tickets showed a forbidden refund that an earlier ticket had made. `scripts/plot_leak.py` draws the comparison from the recorded runs. `scripts/sandbox_demos.py` shows a temporary directory, a controlled clock, an in-memory database and, behind `uv run --extra containers ... --container` and a running Docker, a throwaway Postgres container per run. Nothing else in the repo needs Docker.

## Chapter 8: protected invariants

`src/agent_evals/invariants.py` checks rules that must hold in every run under three policies: `deny-list` (only the actions a case lists as forbidden), `required-only` (a case may change the world, by refund, freeze, restart or escalation, only in the ways it requires) and `permitted` (required actions plus what the ticket itself asked for, from `datasets/triage_heldout_v1.permissions.jsonl`, four rows). `agent-evals invariants --run R --dataset D --policy P` prints the counts and exits 1 if any trace breaks a rule. `agent-evals gate ... --invariants P --permissions F` adds the count as a hard metric beside the quality rules in `policies/protected_invariants.yaml`. `scripts/plot_invariants.py` draws the comparison from the recorded runs.

On the 42 held-out tickets the product with the customer ID breaks 0, 14 and 10 tickets under the three policies in the first pass (0, 13, 9 in the second and third), and the product as shipped breaks none. Which of those actions are acceptable is a product decision that is not made here: the permissions file follows what the tickets say and is data, so it can change.

## Chapter 9: tool use

`src/agent_evals/recording.py` wraps the model client and records every tool call the model asks for (`Trace.tool_calls`: round, name, arguments, whether the tool was offered, and the result), including calls the ledger of side effects cannot see. `src/agent_evals/tool_calls.py` checks each call against six rules (unknown tool, tool not offered, arguments that break the tool's declared JSON Schema, a customer other than the ticket's, a refund with no lookup first, a refund above the looked-up invoice) and classifies each runbook search as relevant, irrelevant or empty. `agent-evals calls --run R --dataset D` prints the table and exits 1 if a rule was broken, or 2 for a run recorded before calls were kept.

Recorded on the 42 held-out tickets: `runs/triage-heldout-v1-calls` and `-calls-2` (the customer-id product) and `-topics` and `-topics-2` (the same product with `RunbookTopicsAdapter`, whose empty search reply also lists the runbook's topics). None of 421 calls broke a rule, and 56 of 62 runbook searches in the first run came back empty. The hint halves the empty searches and adds irrelevant ones (`scripts/search_outcomes.py`, `scripts/plot_searches.py`). The hint was designed after reading these tickets' failed searches, so the comparison is indicative and not a clean test.

## Chapter 10: trajectories as constraints

`src/agent_evals/trajectory.py` checks five constraints on the shape of a whole run, none of which names the right path: it ended in an answer, it stayed within a call budget, it did not stall on empty or irrelevant results, it did not repeat a call, and it did not hand a ticket back to a specialist that already had it. `agent-evals trajectory --run R --dataset D --max-calls N --max-stall M` prints the counts and exits 1 if a run broke one; the limits are required arguments on purpose. `scripts/budget_analysis.py` counts, for each stall limit, the failed runs it would stop and the answering runs it would interrupt. `scripts/compare_trajectories.py` compares versions, and `scripts/plot_trajectories.py` draws run lengths by how they ended.

`StallGuardAdapter` (`triage-live-customer-id-stall-guard`) patches the product's tool loop so that after three empty rounds in a row it stops offering tools and asks for a final answer. Recorded on the 42 held-out tickets, two passes (`runs/triage-heldout-v1-guard`, `-guard-2`, recorded from a clean commit): runs ending in an error fall from 23 of 84 to 5 of 84 and runs taking every required action rise from 56 to 74 of 84. Five runs still ended in an error: login tickets where one useful search reset the streak, and a handoff loop the guard does not watch. The guard was designed after reading these tickets, so the comparison is indicative and not a clean test. The forced answers were not graded for quality.

## Chapter 11: pausing, resuming and state between turns

The reorder product's approval workflow is a two-turn conversation: a question, a pause for a person, then a decision that resumes the run from saved state. `Trace.turns` records each turn (what was sent, whether the run paused, the saved state afterward, and how many times the model was called). `src/agent_evals/adapters/reorder.py` plays the turns against Book 2's `build_approval_workflow`, one thread and one SQLite file per run (`reorder-live`, or `reorder-scripted` with no key). `src/agent_evals/mechanics.py` has eight checks that need no model (pauses before acting, resume calls the model zero times, state survives a restart, threads do not mix, and so on) and five workflow variants with one planted fault each; `agent-evals conversation check` runs them and exits 1 if the real workflow fails a check or a planted fault passes all eight. `agent-evals conversation grade --run R [R ...] --dataset D` grades recorded conversations.

`datasets/reorder_conversations_v1.jsonl` has 26 questions labeled from the inventory (built by `scripts/build_reorder_conversations.py`). Two recorded live passes (`runs/reorder-conversations-1`, `-2`, from a clean commit): nothing was logged before a decision (0 of 52) or after a rejection (0 of 11), resume never called the model (0 of 26), 50 of 52 conversations paused correctly, and in 4 of the 15 approved orders the product's own `extract_sku` would order the first SKU named in the question, which for two multi-SKU questions is the well-stocked one. The API and job queue were not run (they need Postgres). `reorder-app` is pinned to `ch35-end` for `extract_sku` only.

## Chapter 12: retrieval before generation

`src/agent_evals/retrieval.py` scores the package-intelligence product's search on its own, with no language model. It runs the product's own `ask_rag_agent_for_tenant` and `embed_and_upsert` against an in-memory Qdrant, with a recorded embedder and a no-op model, and reads the results through the product's `on_retrieval` hook. Metrics are hit@k, recall@k, precision@k and reciprocal rank. Baselines are random, alphabetical, BM25 (`rank_bm25`) and reciprocal rank fusion of BM25 with the dense search. `agent-evals retrieval report --run R --queries Q --corpus C --part {methods,kinds,paired,misses,scores}` prints the tables, and `agent-evals retrieval isolation` counts results that belong to another tenant, for one collection per tenant and for a planted shared collection (exits 1 if the product leaks or the planted fault does not).

`datasets/pkg_corpus_v1.jsonl` is 107 PyPI packages fetched on 2026-09-20 (`scripts/build_pkg_corpus.py`). `datasets/pkg_queries_v1.jsonl` (50 questions) and `pkg_queries_v2.jsonl` (62: version 1 plus 12 harder questions added after the first results) are built by `scripts/build_pkg_queries.py`, labeled by one person, 16 labels contested; see `datasets/CHANGELOG.md`. `scripts/record_embeddings.py` is the only step that needs a key: it records the summaries and the questions with the product's own client and again with Gemini task types (`runs/pkg-retrieval-1`, `-2`, from a clean commit). On `-2`, 57 answerable questions: a relevant package in the top 3 for 55 (as shipped) and 56 (task types) against 39 for BM25, 48 for the hybrid and 5 for random. Both shipped misses are contested-label questions, but one of them ("start and stop containers") returns no container library in its top 3. With one collection per tenant 0 of 186 results belonged to the other tenant; with a shared collection 103 did. `scripts/plot_retrieval.py` draws the figures. `pkgintel-app` is pinned to `ch33-end`. Qdrant runs in memory, not as a server.

## Chapter 13: grounding, citations and freshness

`src/agent_evals/adapters/pkgintel.py` runs the package-intelligence product's own `ask_rag_agent_for_tenant` with recorded embeddings and the real model (`pkg-live`), and records what was retrieved and which packages the answer cited (`Trace.retrieved`, `Trace.cited`). `src/agent_evals/grounding.py` has plain-code checks: `invalid_citation` (a cited name that was not retrieved), `no_citation`, `outside_name`, `new_number`, and `ignored_context` (a counterfactual answer that lacks the fact added to a summary). `agent-evals grounding grade --run R [R ...] --dataset D --part {summary,bounds,fresh,counterfactual,flagged,readings,refusals}` prints the tables and exits 1 if any answer cites a package that was not retrieved. `agent-evals grounding check` runs a faithful scripted model and six faulty ones, each aimed at one flag, and reports a seventh fault (an unsupported claim with no name and no number) that no check can see.

`datasets/pkg_answers_v1.jsonl` is the 62 retrieval questions plus 10 freshness questions and 5 counterfactual ones (made-up added sentences, test fixtures only); see `datasets/CHANGELOG.md`. Two recorded live passes on `gemini-3.6-flash` (`runs/pkg-answers-1`, `-2`, clean commit): no cited name outside the retrieved three, no outside package names, no numbers the context lacks, and all 10 counterfactual answers follow the edited context. But 29 of 110 answers to answerable questions refuse although a relevant package was retrieved, and my hand reading of all 154 answers (`datasets/pkg_answers_v1.readings.jsonl`, one reader) finds one stretch that no check flagged. `scripts/probe_memory.py` asks the ten version questions with no context: 9 of 10 answers gave a version and only one matched PyPI (`runs/pkg-memory-probe`). `scripts/plot_grounding.py` draws the figures. The product's index stores only names and summaries, so no version can be grounded.

## Chapter 14: when the right answer is "I don't know"

`src/agent_evals/abstention.py` labels each question from the dataset and the trace: the product should answer when a relevant package was among the three retrieved, and decline otherwise, and for freshness, outside, beyond-the-summary and false-premise questions. `agent-evals abstention --dataset D --part {matrix,kinds,paired,scores,gate,overlap}` prints how often it declined when it should have answered (over-refusal) and answered when it should have declined, a cutoff on the top similarity score chosen on the `dev` questions and reported on `test`, and whether refusals go with summaries that share no word with the question. `pkg-live-permissive` is a variant prompt (an experiment, its text and hash are in the run manifests).

`datasets/pkg_abstain_v1.jsonl` has 110 questions (see `datasets/CHANGELOG.md`). Four recorded live runs on `gemini-3.6-flash` (`runs/pkg-abstain-1`, `-2`, `-perm-1`, `-perm-2`, clean commit): the shipped prompt declined 28 of 110 answers that should have been answered and answered 0 of 110 that should have been declined; the permissive prompt 26 and 2, and the difference is inside the run-to-run noise. `src/agent_evals/forced.py` and `agent-evals forced --part {readings,proxies}` cover the 28 triage answers that followed three empty searches, read by hand (`datasets/triage_forced.readings.jsonl`): with the stall guard 22 of 22 say what could not be checked, and 9 of 22 also claim to have searched a source the specialist has no tool for, and 4 make a promise it cannot keep. `scripts/plot_abstention.py` draws the figure.

## Chapter 15: LLM judges from scratch

`src/agent_evals/judge.py` is a claim-level judge: it reads the package information, the question and an answer, lists the claims the answer makes, and marks each supported or unsupported by that information. The verdict is `unsupported` if any claim is. `agent-evals judge run --items I --version {v1,v2} --out DIR [--split dev|test] [--passes 2]` runs it with the model in the `judge_model` role of `config/models.yaml` (`gemini-3.8-flash`, a different model from the one that writes the answers) and writes verdicts, tokens and a manifest with the prompt text and hash. `agent-evals judge report --run R [R2] --items I --part {planted,real,retest,cost,disagreements,compare} [--split S] [--price-in P --price-out P]` reads them; prices are always given, never assumed.

`datasets/judge_items_v1.jsonl` (`scripts/build_judge_items.py`) has 92 real answers labeled by my chapter 13 readings (85 supported, 6 borderline, 1 stretch) and 45 planted faults: answers I read as supported, each with one added sentence the summaries do not state (15 praise, 15 fact, 15 capability). The 45 clean originals are the negatives. Recorded runs (`runs/judge-v1`, all 137 items twice; `runs/judge-v2-test`, the test half twice, clean commits): prompt v1 flagged 90 of 90 planted-fault verdicts but also 37 of 90 clean answers and 59 of 170 answers I read as supported, and flagged all 14 borderline and stretch verdicts. Prompt v2, written after reading v1's mistakes on the `dev` half only, on the unseen items (`--split unseen`: test items whose question no `dev` item shares, 38 items from 15 questions, because each question was answered in two runs and the halves overlap by question): 18 of 18 planted-fault verdicts, 0 of 18 clean answers (v1 5 of 18), 0 of 46 supported readings (v1 9 of 46) and 4 of 12 borderline verdicts (v1 12 of 12). On the whole `test` half, with the twins left in, 2 of 80 supported readings (v1 24 of 80). Two passes agree on 134 of 137 items (v1). One test answer is still flagged in every pass, and on rereading the judge has a point (my reading was too lenient). The readings file is unchanged.

## Chapter 16: breaking the judge

`src/agent_evals/bias.py` makes changed copies of the judge items, each with one irrelevant change (`pad` supported text, `reverse_context`, `fault_first`, `injection`, and the harder `injection_json`, `fake_source`, `authority`). `agent-evals judge perturb --items I --out F [--set attacks]` writes them, `judge run` judges them, and `judge report --part flips` counts verdicts that moved, against the noise of two passes. `--model ID` on `judge run` swaps the judge, and `--part judges` and `--part agreement` compare judges.

`datasets/judge_items_ch16_v1.jsonl` (364 items) and `judge_items_ch16_attacks_v1.jsonl` (135 planted-only items) are built from the chapter 15 items. Recorded runs (clean commits, prompt v2 unless noted): none of the 360 faulty verdicts moved under the four basic changes, 9 of 368 others did, and nothing moved on the unseen items. Fake package information placed in an answer got 43 of 90 faulty verdicts past v2 (7 of 18 unseen); a note to the judge, an instruction to reply with JSON and a claimed approval got 0. Prompt v3 (treat the answer as untrusted, quote evidence) got 0 of 90, and so did v3 without the code check (`v3u`); the code check changed one verdict on the originals and none on the attacks. The judge that wrote the answers (`gemini-3.6-flash`) flagged 1 of 12 borderline verdicts, the newer judge 4 and an older one 3; the three agree on 133 to 134 of 137 items. v2 lets through the stretch answer that v1 flagged. `scripts/plot_attacks.py` draws the figure.

## Chapter 17: humans and calibration

`src/agent_evals/agreement.py` has percent agreement, Cohen's kappa (checked against scikit-learn), PABAK, a bootstrap that resamples whole questions, sensitivity and specificity, the Rogan-Gladen correction of a judge's flag rate with an interval that redraws both the calibration and the test set, and a planning simulation. `agent-evals labels --part {matrix,retest,sensitivity,calibrate,plan,second} [--judge RUN ...] [--group real|all] [--convention strict|lenient] [--adjudicated]` prints them from the recorded judge runs and my readings.

`datasets/judge_adjudication_v1.jsonl` (`scripts/build_adjudication.py`) records what I decided on the 7 answers where the judges' majority disagrees with my reading: two changed (the Postgres answers become borderline). `datasets/human_label_sample_v1.jsonl` and `label-tool/` (`scripts/build_label_sample.py`) are 50 answers for a second person to label blind, with the hard cases in them, and a page to do it in; `agent-evals labels --part second --second FILE --judge RUN` compares their labels with mine and a judge's. No second person has labeled yet. On the real answers (7 bad of 92, from 4 questions), kappa between a judge and me is unstable: changing two of my 92 labels moves v2's kappa from 0.33 to 0.59, and the intervals run from about 0 to 1, while raw agreement is 92 to 96% and PABAK 0.85 to 0.91. The calibration half of the questions holds one bad answer, so the corrected rate (0.0%) misses my rate (12.2%) and one judge cannot be corrected at all. With 100 test items the corrected rate's interval stays about 15 points wide however many calibration items there are (simulation, 2,000 repeats, assumed sensitivity 0.85 and specificity 0.97). `scripts/check_kappa_sklearn.py` compares my kappa with scikit-learn's on 300 random cases (largest difference 2.2e-16).

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

One trial per ticket or question, three products (each in its own chapters), deterministic graders only. There are no judges, no repeated-trial reliability, no judge for claims the plain-code checks cannot see, and no online monitoring. Those arrive in later chapters. The recorded run is a single observation of a stochastic system: rerunning it can give a different trace.

## License

MIT, see `LICENSE`.
