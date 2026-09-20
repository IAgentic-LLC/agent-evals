# Scorecard: triage-heldout-v1-guard

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id-stall-guard.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 40/42 (95.2%)    | 84.2% to 98.7%           |
| Required actions taken  | 38/42 (90.5%)    | 77.9% to 96.2%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 2                |                          |
| Latency median / p95    | 7.70 s / 15.38 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
