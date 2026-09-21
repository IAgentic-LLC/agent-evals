# Scorecard: triage-change-topics

Dataset `triage_change_v1`: 40 cases, 120 observations
(3 trial(s) each). Adapter: triage-live-customer-id-topics.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 77/120 (64.2%)   | 55.3% to 72.2%           |
| Required actions taken  | 77/120 (64.2%)   | 55.3% to 72.2%           |
| Forbidden actions taken | 0/120            | upper bound 3.1% of runs |
| Errors                  | 43               |                          |
| Latency median / p95    | 7.94 s / 17.86 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
