# Scorecard: triage-heldout-v1-calls-2

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 31/42 (73.8%)    | 58.9% to 84.7%           |
| Required actions taken  | 29/42 (69.0%)    | 54.0% to 80.9%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 11               |                          |
| Latency median / p95    | 7.57 s / 17.75 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
