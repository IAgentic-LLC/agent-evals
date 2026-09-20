# Scorecard: triage-heldout-v1-shipped

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 33/42 (78.6%)    | 64.1% to 88.3%           |
| Required actions taken  | 19/42 (45.2%)    | 31.2% to 60.1%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 9                |                          |
| Latency median / p95    | 6.60 s / 16.58 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
