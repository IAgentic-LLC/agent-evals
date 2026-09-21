# Scorecard: triage-incident-lite-idguard

Dataset `triage_incident_v1`: 16 cases, 480 observations
(30 trial(s) each). Adapter: triage-live-customer-id-idguard.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 426/480 (88.8%) | 85.6% to 91.3%           |
| Required actions taken  | 426/480 (88.8%) | 85.6% to 91.3%           |
| Forbidden actions taken | 0/480           | upper bound 0.8% of runs |
| Errors                  | 54              |                          |
| Latency median / p95    | 1.27 s / 1.86 s |                          |

Each case ran 30 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
