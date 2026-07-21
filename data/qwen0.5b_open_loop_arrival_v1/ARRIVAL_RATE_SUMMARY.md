# Fixed-rate open-loop arrival sweep

Requests are injected at fixed wall-clock arrival times through
LLMEngine.add_request. This is an open-loop workload; overdue arrivals
retain their scheduled arrival timestamp so queueing is included in TTFT.

| Offered req/s | Method | Achieved req/s | Handler ms | TTFT p95 | TTFT p99 | E2E p95 | E2E p99 | TTFT SLO | E2E SLO | Quality |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 8.000 | async_raw | 7.920 | 40.036 | 96.987 | 101.744 | 230.645 | 235.422 | 100.0% | 100.0% | pass |
| 8.000 | async_cuszp_1e-5 | 7.981 | 29.810 | 69.949 | 79.849 | 221.178 | 225.852 | 100.0% | 100.0% | pass |

## 5-trial paired differences

Differences are cuSZp minus raw; intervals are paired 95% CI
half-widths. Negative latency and positive throughput are better.

| Offered req/s | Handler diff ms | Achieved req/s diff | Mean initial E2E diff ms |
|---:|---:|---:|---:|
| 8.000 | -10.226 +/- 0.626 | 0.061 +/- 0.017 | -36.982 +/- 12.829 |
