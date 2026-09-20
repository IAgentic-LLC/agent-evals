# Scorecard: triage-heldout-v1-shipped-2

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 32/42 (76.2%)    | 61.5% to 86.5%           |
| Required actions taken  | 18/42 (42.9%)    | 29.1% to 57.8%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 10               |                          |
| Latency median / p95    | 5.93 s / 15.48 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
