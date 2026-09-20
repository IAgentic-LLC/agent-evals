# Scorecard: triage-heldout-v1-topics

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id-topics.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 35/42 (83.3%)    | 69.4% to 91.7%           |
| Required actions taken  | 33/42 (78.6%)    | 64.1% to 88.3%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 7                |                          |
| Latency median / p95    | 7.83 s / 17.88 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
