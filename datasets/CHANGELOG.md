# Dataset changelog

Every published version of a dataset is frozen. A change is a new version with an entry here, never an edit in place.

## triage_heldout_v2 (2026-09-20)

Made from `triage_heldout_v1` by `scripts/make_heldout_v2.py`. Nothing else changed: same 42 cases, same tickets, same expected specialists.

- **HO-026, HO-041** (lost phone): `required_actions` changed from `escalate_to_oncall` to `freeze_account`. The customer asked for the account to be locked, so locking it is the required work. Version 1 required escalation for every security ticket, which was a label by specialist and not by ticket. Found when reading the first run's four security tickets that took no escalation.
- **HO-027, HO-035** (password in a breach, "is this email real?"): labels unchanged, marked `label_confidence: contested`. Two experts could reasonably disagree on whether escalation is required. Needs a second labeler.
- New slice `label_confidence` on every case: `clear` (40 cases) or `contested` (2 cases).

Caution: these changes were made after seeing one run's outputs. That is a way to fit labels to a system's behavior. The two label changes are justified by the ticket text alone. Judge them yourself: the version 1 file and its recorded run are kept, so the original scores can always be recomputed.

## triage_heldout_v1 (2026-09-20)

42 tickets written by the author. Frozen before the first model run. See `triage_heldout.card.md`.
