# Scorecard: triage-metered-3-6-guard

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id-stall-guard.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 118/126 (93.7%)  | 88.0% to 96.7%           |
| Required actions taken  | 111/126 (88.1%)  | 81.3% to 92.7%           |
| Forbidden actions taken | 0/126            | upper bound 3.0% of runs |
| Errors                  | 8                |                          |
| Latency median / p95    | 8.25 s / 16.36 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
