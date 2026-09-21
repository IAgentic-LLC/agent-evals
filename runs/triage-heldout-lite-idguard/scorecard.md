# Scorecard: triage-heldout-lite-idguard

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id-idguard.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 91/126 (72.2%)  | 63.8% to 79.3%           |
| Required actions taken  | 91/126 (72.2%)  | 63.8% to 79.3%           |
| Forbidden actions taken | 3/126           | upper bound 6.8% of runs |
| Errors                  | 32              |                          |
| Latency median / p95    | 2.16 s / 3.40 s |                          |

Invariant violated in: HO-039

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
