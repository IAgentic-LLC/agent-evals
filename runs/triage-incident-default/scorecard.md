# Scorecard: triage-incident-default

Dataset `triage_incident_v1`: 16 cases, 160 observations
(10 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 160/160 (100.0%) | 97.7% to 100.0%          |
| Required actions taken  | 160/160 (100.0%) | 97.7% to 100.0%          |
| Forbidden actions taken | 0/160            | upper bound 2.3% of runs |
| Errors                  | 0                |                          |
| Latency median / p95    | 3.46 s / 6.55 s  |                          |

Each case ran 10 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
