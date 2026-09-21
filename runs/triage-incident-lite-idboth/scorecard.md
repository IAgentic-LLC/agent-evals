# Scorecard: triage-incident-lite-idboth

Dataset `triage_incident_v1`: 16 cases, 480 observations
(30 trial(s) each). Adapter: triage-live-customer-id-idboth.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 480/480 (100.0%) | 99.2% to 100.0%          |
| Required actions taken  | 480/480 (100.0%) | 99.2% to 100.0%          |
| Forbidden actions taken | 0/480            | upper bound 0.8% of runs |
| Errors                  | 0                |                          |
| Latency median / p95    | 1.26 s / 1.47 s  |                          |

Each case ran 30 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
