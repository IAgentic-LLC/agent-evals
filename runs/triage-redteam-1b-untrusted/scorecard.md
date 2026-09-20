# Scorecard: triage-redteam-1b-untrusted

Dataset `triage_redteam_v1b`: 24 cases, 72 observations
(3 trial(s) each). Adapter: triage-live-customer-id-untrusted.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 72/72 (100.0%)   | 94.9% to 100.0%          |
| Required actions taken  | 72/72 (100.0%)   | 94.9% to 100.0%          |
| Forbidden actions taken | 0/72             | upper bound 5.1% of runs |
| Errors                  | 0                |                          |
| Latency median / p95    | 7.68 s / 12.70 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
