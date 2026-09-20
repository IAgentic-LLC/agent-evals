# Scorecard: triage-reliability-5x

Dataset `triage_heldout_v1`: 42 cases, 210 observations
(5 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 148/210 (70.5%)  | 64.0% to 76.2%           |
| Required actions taken  | 137/210 (65.2%)  | 58.6% to 71.4%           |
| Forbidden actions taken | 0/210            | upper bound 1.8% of runs |
| Errors                  | 62               |                          |
| Latency median / p95    | 8.69 s / 18.35 s |                          |

Each case ran 5 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
