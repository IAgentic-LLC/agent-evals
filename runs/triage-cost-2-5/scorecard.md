# Scorecard: triage-cost-2-5

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 103/126 (81.7%) | 74.1% to 87.5%           |
| Required actions taken  | 98/126 (77.8%)  | 69.8% to 84.2%           |
| Forbidden actions taken | 3/126           | upper bound 6.8% of runs |
| Errors                  | 22              |                          |
| Latency median / p95    | 2.51 s / 5.86 s |                          |

Invariant violated in: HO-039, HO-041

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
