# Scorecard: triage-redteam-2b-untrusted

Dataset `triage_redteam_v2b`: 4 cases, 12 observations
(3 trial(s) each). Adapter: triage-live-customer-id-untrusted.

| Measure                 | Observed        | 95% interval              |
|-------------------------|-----------------|---------------------------|
| Routing correct         | 12/12 (100.0%)  | 75.8% to 100.0%           |
| Required actions taken  | 12/12 (100.0%)  | 75.8% to 100.0%           |
| Forbidden actions taken | 0/12            | upper bound 24.2% of runs |
| Errors                  | 0               |                           |
| Latency median / p95    | 7.67 s / 9.00 s |                           |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
