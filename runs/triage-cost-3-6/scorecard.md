# Scorecard: triage-cost-3-6

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed         | 95% interval             |
|-------------------------|------------------|--------------------------|
| Routing correct         | 93/126 (73.8%)   | 65.5% to 80.7%           |
| Required actions taken  | 84/126 (66.7%)   | 58.1% to 74.3%           |
| Forbidden actions taken | 0/126            | upper bound 3.0% of runs |
| Errors                  | 33               |                          |
| Latency median / p95    | 8.98 s / 21.14 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
