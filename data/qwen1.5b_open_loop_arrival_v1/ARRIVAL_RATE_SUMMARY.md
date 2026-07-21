# Fixed-rate open-loop arrival sweep

Requests are injected at fixed wall-clock arrival times through
LLMEngine.add_request. This is an open-loop workload; overdue arrivals
retain their scheduled arrival timestamp so queueing is included in TTFT.

| Offered req/s | Method | Achieved req/s | Handler ms | TTFT p95 | TTFT p99 | E2E p95 | E2E p99 | TTFT SLO | E2E SLO | Quality |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2.000 | async_raw | 2.059 | 71.601 | 174.358 | 176.797 | 286.363 | 289.641 | 100.0% | 100.0% | pass |
| 2.000 | async_cuszp_1e-5 | 2.073 | 57.985 | 141.609 | 144.379 | 256.855 | 259.653 | 100.0% | 100.0% | pass |
| 4.000 | async_raw | 3.757 | 71.038 | 394.132 | 403.257 | 503.609 | 512.619 | 100.0% | 100.0% | pass |
| 4.000 | async_cuszp_1e-5 | 4.032 | 57.618 | 143.755 | 161.847 | 257.821 | 264.437 | 100.0% | 100.0% | pass |
| 6.000 | async_raw | 3.751 | 71.985 | 1631.349 | 1661.501 | 1738.804 | 1774.099 | 25.0% | 36.2% | pass |
| 6.000 | async_cuszp_1e-5 | 4.415 | 57.867 | 1005.363 | 1068.656 | 1102.684 | 1156.893 | 47.5% | 65.0% | pass |

## Five-trial paired differences

Differences are cuSZp minus raw; intervals are paired 95% CI
half-widths. Negative latency and positive throughput are better.

| Offered req/s | Handler diff ms | Achieved req/s diff | Mean initial E2E diff ms |
|---:|---:|---:|---:|
| 2.000 | -13.615 +/- 0.822 | 0.015 +/- 0.002 | -41.878 +/- 1.533 |
| 4.000 | -13.419 +/- 1.758 | 0.275 +/- 0.010 | -150.268 +/- 10.044 |
| 6.000 | -14.117 +/- 1.671 | 0.665 +/- 0.084 | -357.525 +/- 54.074 |
