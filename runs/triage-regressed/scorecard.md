# Scorecard: triage-regressed

Dataset `triage_book3_six`: 6 cases, 6 observations
(1 trial(s) each). Adapter: triage-regressed-scripted.

| Measure                 | Observed        | 95% interval              |
|-------------------------|-----------------|---------------------------|
| Routing correct         | 6/6 (100.0%)    | 61.0% to 100.0%           |
| Required actions taken  | 0/6 (0.0%)      | 0.0% to 39.0%             |
| Forbidden actions taken | 2/6             | upper bound 70.0% of runs |
| Errors                  | 0               |                           |
| Latency median / p95    | 0.00 s / 0.00 s |                           |

Invariant violated in: TCK-1003, TCK-1005

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
