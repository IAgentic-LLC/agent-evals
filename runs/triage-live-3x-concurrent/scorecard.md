# Scorecard: triage-live-3x-concurrent

Dataset `triage_book3_six`: 6 cases, 18 observations
(3 trial(s) each). Adapter: triage-live.

| Measure                 | Observed        | 95% interval              |
|-------------------------|-----------------|---------------------------|
| Routing correct         | 18/18 (100.0%)  | 82.4% to 100.0%           |
| Required actions taken  | 12/18 (66.7%)   | 43.7% to 83.7%            |
| Forbidden actions taken | 0/18            | upper bound 17.6% of runs |
| Errors                  | 0               |                           |
| Latency median / p95    | 6.53 s / 9.94 s |                           |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
