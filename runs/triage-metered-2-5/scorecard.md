# Scorecard: triage-metered-2-5

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 104/126 (82.5%) | 75.0% to 88.2%           |
| Required actions taken  | 95/126 (75.4%)  | 67.2% to 82.1%           |
| Forbidden actions taken | 1/126           | upper bound 4.4% of runs |
| Errors                  | 22              |                          |
| Latency median / p95    | 2.26 s / 6.19 s |                          |

Invariant violated in: HO-041

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
