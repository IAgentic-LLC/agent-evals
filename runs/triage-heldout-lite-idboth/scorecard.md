# Scorecard: triage-heldout-lite-idboth

Dataset `triage_heldout_v1`: 42 cases, 126 observations
(3 trial(s) each). Adapter: triage-live-customer-id-idboth.

| Measure                 | Observed        | 95% interval             |
|-------------------------|-----------------|--------------------------|
| Routing correct         | 102/126 (81.0%) | 73.2% to 86.9%           |
| Required actions taken  | 99/126 (78.6%)  | 70.6% to 84.8%           |
| Forbidden actions taken | 0/126           | upper bound 3.0% of runs |
| Errors                  | 23              |                          |
| Latency median / p95    | 2.03 s / 3.25 s |                          |

Each case ran 3 times. Trials of one case are not independent,
so read these intervals as describing these trials, not new tickets.

There is no combined score on purpose. Each measure carries its own
uncertainty, and the invariant is a count that must be zero, not an
average.
