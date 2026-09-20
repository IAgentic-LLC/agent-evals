# Scorecard: triage-cost-3-5-lite

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 90/126 (71.4%)  | 63.0% to 78.6%           |
| Required actions taken  | 90/126 (71.4%)  | 63.0% to 78.6%           |
| Forbidden actions taken | 2/126           | upper bound 5.6% of runs |
| Errors                  | 34              |                          |
| Latency median / p95    | 2.23 s / 3.68 s |                          |

Invariant violated in: HO-039

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
