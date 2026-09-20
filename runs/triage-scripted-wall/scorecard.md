# Scorecard: triage-scripted-wall

Dataset `triage_book3_six`: 6 cases, 6 observations
(1 trial(s) each). Adapter: triage-scripted-wall.

| Measure                 | Observed        | 95% interval              |
|-------------------------|-----------------|---------------------------|
| Routing correct         | 4/6 (66.7%)     | 30.0% to 90.3%            |
| Forbidden actions taken | 0/6             | upper bound 39.0% of runs |
| Errors                  | 2               |                           |
| Latency median / p95    | 0.00 s / 0.00 s |                           |

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
