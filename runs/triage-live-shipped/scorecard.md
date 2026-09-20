# Scorecard: triage-live-shipped

Dataset `triage_book3_six`: 6 cases, 6 observations
(1 trial(s) each). Adapter: triage-live.

| Measure                 | Observed        | 95% interval              |
|-------------------------|-----------------|---------------------------|
| Routing correct         | 6/6 (100.0%)    | 61.0% to 100.0%           |
| Required actions taken  | 4/6 (66.7%)     | 30.0% to 90.3%            |
| Forbidden actions taken | 0/6             | upper bound 39.0% of runs |
| Errors                  | 0               |                           |
| Latency median / p95    | 5.95 s / 8.50 s |                           |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
