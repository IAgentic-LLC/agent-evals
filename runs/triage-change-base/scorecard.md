# Scorecard: triage-change-base

Dataset `triage_change_v1`: 40 cases, 120 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed          | 95% interval             |
|-------------------------|-------------------|--------------------------|
| Routing correct         | 28/120 (23.3%)    | 16.7% to 31.7%           |
| Required actions taken  | 28/120 (23.3%)    | 16.7% to 31.7%           |
| Forbidden actions taken | 0/120             | upper bound 3.1% of runs |
| Errors                  | 92                |                          |
| Latency median / p95    | 17.56 s / 36.04 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
