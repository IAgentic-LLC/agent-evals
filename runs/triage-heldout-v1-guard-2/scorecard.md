# Scorecard: triage-heldout-v1-guard-2

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id-stall-guard.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 39/42 (92.9%)    | 81.0% to 97.5%           |
| Required actions taken  | 36/42 (85.7%)    | 72.2% to 93.3%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 3                |                          |
| Latency median / p95    | 8.13 s / 15.58 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
