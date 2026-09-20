# Dataset changelog

Every published version of a dataset is frozen. A change is a new version with an entry here, never an edit in place.

## judge_adjudication_v1 and human_label_sample_v1 (2026-09-20)

`judge_adjudication_v1` is made by `scripts/build_adjudication.py` for chapter 17: my decision on each of the 7 real answers where the majority of five judge runs over both passes disagrees with my reading. Two change (`pkg-answers-1:PQ-013` and `pkg-answers-2:PQ-013`, supported to borderline, because the summary of SQLAlchemy does not mention Postgres). The readings file itself is not changed. **Caution:** the same person made the readings and the adjudication, after seeing the judges, which is the weakest adjudication there is.

`human_label_sample_v1` is made by `scripts/build_label_sample.py` (seed 1): 50 items from `judge_items_v1` for a second person to label blind: 7 I read as borderline or stretch, 4 I read as supported that a judge flagged, 31 supported (one per question) and 8 planted faults. The page `label-tool/index.html` shows the question, the package information and the answer, and no label. The stratum column is for the analysis and is not shown to the labeler.

## judge_items_ch16_v1 and judge_items_ch16_attacks_v1 (2026-09-20)

Made by `agent-evals judge perturb` (`src/agent_evals/bias.py`) for chapter 16, from `judge_items_v1`. Each item is a copy of one original with one change and carries `base_id` and `perturbation`. `_v1` has 364 items: `pad` (137), `reverse_context` (137), `fault_first` (45), `injection` (45). `_attacks_v1` has 135, planted answers only: `injection_json`, `fake_source`, `authority` (45 each). The changes leave what the answer claims alone, apart from the injected notes, and `fake_source` puts the planted sentence in a line that imitates package information. The splits are inherited from the originals, so `--split unseen` still works.

## judge_items_v1 (2026-09-20)

Made by `scripts/build_judge_items.py` for chapter 15, from the recorded answers of chapter 13 and my readings of them. 137 items: 92 `real` (the answers that cite something; label from my reading: 85 supported, 6 borderline, 1 stretch) and 45 `planted`. A planted item is an answer I read as supported (counterfactual answers excluded), copied with one sentence added that the retrieved summaries do not state: 15 `praise`, 15 `fact`, 15 `capability`. The 45 originals carry `clean_of_planted` and are the negatives. `real` items alternate `dev` and `test`; a planted copy is in the same half as its original, with the kinds cycling within each half. Random choices use seed 0.

- **Caution:** the planted sentences are blunt, and none names a package or contains a number, so plain-code checks cannot see them by design. They test whether a judge sees a claim the code cannot. They do not test subtle cases. The real items do, and their labels are one person's reading.
- **Split caveat:** each question was answered in both runs, so the same question can be in `dev` and `test` (17 of the 46 real test items share a question with a `dev` item). Reports take `--split unseen` for the 38 test items whose question no `dev` item shares (15 questions). A question-level split would be a new version.
- **Known label problem:** `pkg-answers-1:PQ-013` ("You can also use SQLAlchemy" to connect to Postgres) is labeled supported, and a judge flagged it in every pass. On rereading, the summary "Database Abstraction Library" does not say Postgres, so a stricter label would be borderline. The file is frozen so results stay comparable.

## pkg_abstain_v1 and the forced-answer readings (2026-09-20)

`pkg_abstain_v1` is made by `scripts/build_pkg_abstain.py` for chapter 14. 110 cases: the 72 non-counterfactual questions of `pkg_answers_v1` (62 retrieval questions, 10 freshness questions) and 38 new ones, `AB-001` to `AB-038`: 15 `outside` (no package in the index does the job), 15 `beyond_summary` (a package is named and the question asks for a fact its one-line summary lacks) and 8 `false_premise`.

- **Split:** within each kind the cases alternate `dev`, `test` in file order, 57 and 53. The 72 older questions were all `test` before; this file reassigns them. They had already been read closely in chapters 12 and 13, so `dev` is not a clean set for them.
- **Caution:** I wrote the 38 new questions after reading the chapter 13 answers, so I knew the model declines well. The labels (what should be declined) come from one person.
- Embeddings for the new questions are in `runs/pkg-abstain-embeddings`.
- **Arguable labels:** AB-002 (recognize speech; transformers names audio), AB-005 (barcodes; opencv-python) and AB-017 (install polars; pip was retrieved) could be read as answerable. AB-024 and AB-027 never retrieve the package they name, so declining them is right for a different reason. The file is frozen; a corrected version would be a new one.
- `triage_forced.readings.jsonl` holds my reading of the 28 triage answers that followed three empty runbook searches (chapter 10): `gap_stated`, `false_action_claim` and `claims_source_it_lacks`. One reader, the author. `scripts/build_forced_readings.py` holds the rubric.

## pkg_answers_v1 and its readings (2026-09-20)

Made by `scripts/build_pkg_answers.py` for chapter 13. 77 cases: the 62 questions of `pkg_queries_v2` byte for byte (same IDs), then 10 freshness questions (`FQ-001` to `FQ-010`, "what is the latest version of X") and 5 counterfactual questions (`CF-001` to `CF-005`).

- **Freshness:** the expected version is the one PyPI reported on 2026-09-20, kept in `pkg_corpus_v1`. The product's index keeps only a package's name and one-line summary, so no version can come from it.
- **Counterfactual:** each case carries `input.summary_edits`, a made-up sentence added to one package's summary, and is asked against a copy of the index with the edit. The edits are test fixtures. They are not statements about the real packages (for example, requests is not built on a Rust core).
- **Readings:** `pkg_answers_v1.readings.jsonl` holds my reading of each of the 154 recorded answers (two runs of 77): refusal, supported, stretch or unsupported. One reader, the author, no second opinion. `scripts/build_readings.py` says how each answer was assigned.
- Retrieval for two of the freshness questions (pandas, pydantic) does not find the package, because the package name is not embedded.

## pkg_queries_v2 (2026-09-20)

Made from `pkg_queries_v1` by `scripts/build_pkg_queries.py`. The first 50 questions are byte for byte the version 1 questions, in the same order, with the same IDs. Twelve harder questions were added (`PQ-051` to `PQ-062`, kind `hard`), 62 in all.

- **Why:** the recorded version 1 run showed the product's retriever finding a relevant package in the top 3 for 44 of 45 answerable questions, too easy to show a difference between methods. The new questions describe a situation in the user's own words instead of naming the job.
- **Caution:** these twelve were written after seeing the version 1 results. That biases the set toward what the shipped retriever might get wrong, and it is disclosed in the chapter. Eight of the twelve labels are contested.
- The version 1 file and its recorded run (`runs/pkg-retrieval-1`) are kept, so the original scores can always be recomputed.

## pkg_queries_v1 and pkg_corpus_v1 (2026-09-20)

- **Corpus:** 107 PyPI packages in 16 topic groups, fetched on 2026-09-20 with `scripts/build_pkg_corpus.py` through the product's own metadata fetcher. Each row has the name, the version at fetch time, the one-line summary that the product embeds, and the topic used only to pick the packages. The summaries are PyPI metadata written by each project. The license of the file as a whole is an open question for the author.
- **Queries:** 50 questions written by the author after reading the corpus summaries: 40 `task`, 5 `named`, 5 `none` (no relevant package on purpose). Relevance follows what a package does, not the words in its summary. Eight labels are contested. One labeler.

## triage_heldout_v2 (2026-09-20)

Made from `triage_heldout_v1` by `scripts/make_heldout_v2.py`. Nothing else changed: same 42 cases, same tickets, same expected specialists.

- **HO-026, HO-041** (lost phone): `required_actions` changed from `escalate_to_oncall` to `freeze_account`. The customer asked for the account to be locked, so locking it is the required work. Version 1 required escalation for every security ticket, which was a label by specialist and not by ticket. Found when reading the first run's four security tickets that took no escalation.
- **HO-027, HO-035** (password in a breach, "is this email real?"): labels unchanged, marked `label_confidence: contested`. Two experts could reasonably disagree on whether escalation is required. Needs a second labeler.
- New slice `label_confidence` on every case: `clear` (40 cases) or `contested` (2 cases).

Caution: these changes were made after seeing one run's outputs. That is a way to fit labels to a system's behavior. The two label changes are justified by the ticket text alone. Judge them yourself: the version 1 file and its recorded run are kept, so the original scores can always be recomputed.

## triage_heldout_v1 (2026-09-20)

42 tickets written by the author. Frozen before the first model run. See `triage_heldout.card.md`.
