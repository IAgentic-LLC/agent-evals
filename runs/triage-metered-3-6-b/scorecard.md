# Scorecard: triage-metered-3-6-b

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id.

| Measure                 | Observed          | 95% interval             |
|-------------------------|-------------------|--------------------------|
| Routing correct         | 86/126 (68.3%)    | 59.7% to 75.7%           |
| Required actions taken  | 78/126 (61.9%)    | 53.2% to 69.9%           |
| Forbidden actions taken | 0/126             | upper bound 3.0% of runs |
| Errors                  | 40                |                          |
| Latency median / p95    | 10.77 s / 19.97 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
