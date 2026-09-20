# Dataset changelog

Every published version of a dataset is frozen. A change is a new version with an entry here, never an edit in place.

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
