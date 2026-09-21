# Scorecard: triage-heldout-lite-idnote

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id-idnote.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 105/126 (83.3%) | 75.9% to 88.8%           |
| Required actions taken  | 100/126 (79.4%) | 71.5% to 85.5%           |
| Forbidden actions taken | 0/126           | upper bound 3.0% of runs |
| Errors                  | 19              |                          |
| Latency median / p95    | 1.94 s / 3.10 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
