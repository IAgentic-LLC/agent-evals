# Scorecard: triage-heldout-v1-2

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 27/42 (64.3%)    | 49.2% to 77.0%           |
| Required actions taken  | 25/42 (59.5%)    | 44.5% to 73.0%           |
| Forbidden actions taken | 0/42             | upper bound 8.4% of runs |
| Errors                  | 15               |                          |
| Latency median / p95    | 6.22 s / 16.38 s |                          |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
