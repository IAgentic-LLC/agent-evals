# Scorecard: triage-routing-3x

Dataset `triage_routing_v1`: 24 cases, 72 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed          | 95% interval             |
|-------------------------|-------------------|--------------------------|
| Routing correct         | 41/72 (56.9%)     | 45.4% to 67.7%           |
| Required actions taken  | 41/72 (56.9%)     | 45.4% to 67.7%           |
| Forbidden actions taken | 0/72              | upper bound 5.1% of runs |
| Errors                  | 31                |                          |
| Latency median / p95    | 12.59 s / 23.58 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
