# Scorecard: triage-cost-3-6-guard

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id-stall-guard.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 122/126 (96.8%)  | 92.1% to 98.8%           |
| Required actions taken  | 114/126 (90.5%)  | 84.1% to 94.5%           |
| Forbidden actions taken | 0/126            | upper bound 3.0% of runs |
| Errors                  | 4                |                          |
| Latency median / p95    | 8.80 s / 15.72 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
