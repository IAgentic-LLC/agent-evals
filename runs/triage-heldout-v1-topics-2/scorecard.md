# Scorecard: triage-heldout-v1-topics-2

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id-topics.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 37/42 (88.1%)    | 75.0% to 94.8%           |
| Required actions taken  | 35/42 (83.3%)    | 69.4% to 91.7%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 5                |                          |
| Latency median / p95    | 7.65 s / 15.38 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
