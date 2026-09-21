# Scorecard: triage-metered-3-6-c

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 88/126 (69.8%)   | 61.3% to 77.2%           |
| Required actions taken  | 80/126 (63.5%)   | 54.8% to 71.4%           |
| Forbidden actions taken | 0/126            | upper bound 3.0% of runs |
| Errors                  | 38               |                          |
| Latency median / p95    | 9.10 s / 21.05 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
