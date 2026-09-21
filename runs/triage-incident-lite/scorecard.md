# Scorecard: triage-incident-lite

Dataset `triage_incident_v1`: 16 cases, 480 observations
(30 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 424/480 (88.3%) | 85.2% to 90.9%           |
| Required actions taken  | 424/480 (88.3%) | 85.2% to 90.9%           |
| Forbidden actions taken | 0/480           | upper bound 0.8% of runs |
| Errors                  | 56              |                          |
| Latency median / p95    | 1.26 s / 1.89 s |                          |

Each case ran 30 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
