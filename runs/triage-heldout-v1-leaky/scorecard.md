# Scorecard: triage-heldout-v1-leaky

Dataset `triage_heldout_v1`: 42 cases, 42 observations
(1 trial(s) each). Adapter: triage-live-customer-id-leaky.

| Measure                 | Observed         | 95% interval              |
|-------------------------|------------------|---------------------------|
| Routing correct         | 31/42 (73.8%)    | 58.9% to 84.7%            |
| Required actions taken  | 31/42 (73.8%)    | 58.9% to 84.7%            |
| Forbidden actions taken | 5/42             | upper bound 25.0% of runs |
| Errors                  | 11               |                           |
| Latency median / p95    | 6.63 s / 13.87 s |                           |

Invariant violated in: HO-037, HO-038, HO-039, HO-041, HO-042

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
