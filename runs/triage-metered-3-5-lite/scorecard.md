# Scorecard: triage-metered-3-5-lite

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 88/126 (69.8%)  | 61.3% to 77.2%           |
| Required actions taken  | 88/126 (69.8%)  | 61.3% to 77.2%           |
| Forbidden actions taken | 3/126           | upper bound 6.8% of runs |
| Errors                  | 35              |                          |
| Latency median / p95    | 2.02 s / 3.22 s |                          |

Invariant violated in: HO-039

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
