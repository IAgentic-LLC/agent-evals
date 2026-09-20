# Scorecard: triage-heldout-v1-calls

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 30/42 (71.4%)    | 56.4% to 82.8%           |
| Required actions taken  | 27/42 (64.3%)    | 49.2% to 77.0%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 12               |                          |
| Latency median / p95    | 7.83 s / 16.49 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
