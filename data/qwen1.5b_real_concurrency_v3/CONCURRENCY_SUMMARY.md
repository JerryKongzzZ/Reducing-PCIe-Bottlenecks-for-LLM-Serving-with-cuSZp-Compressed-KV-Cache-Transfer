# Real-request concurrency sweep

No synthetic PCIe copy contender is used. Every level submits unique
long-context requests in one vLLM batch. V1 model runner is forced
because vLLM marks UVA unavailable under WSL.

| Concurrency | Method | Ratio | G2C ms | C2G ms | Total ms | Initial req/s | Replay req/s | Replay TTFT ms | Quality |
|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 2 | async_raw | 1.00000x | 69.909 | 2.037 | 71.946 | 3.764 | 13.731 | 60.272 | pass |
| 2 | async_cuszp_1e-5 | 1.61571x | 58.349 | 2.451 | 60.799 | 4.478 | 13.673 | 61.892 | pass |
| 4 | async_raw | 1.00000x | 69.666 | 2.694 | 72.360 | 3.767 | 12.730 | 139.467 | pass |
| 4 | async_cuszp_1e-5 | 1.61607x | 52.192 | 2.562 | 54.754 | 4.580 | 13.233 | 133.552 | pass |
| 8 | async_raw | 1.00000x | 69.931 | 2.416 | 72.347 | 3.758 | 13.129 | 280.671 | pass |
| 8 | async_cuszp_1e-5 | 1.61590x | 51.221 | 2.642 | 53.863 | 4.534 | 13.194 | 277.638 | pass |
| 16 | async_raw | 1.00000x | 69.585 | 2.633 | 72.218 | 3.729 | 12.881 | 586.986 | pass |
| 16 | async_cuszp_1e-5 | 1.61590x | 56.205 | 2.526 | 58.730 | 4.433 | 13.494 | 555.891 | pass |

## Pooled burst tail latency and SLO

Tails pool all requests across the five trials before computing
percentiles. These are simultaneous-batch burst results, not an
open-loop arrival-rate experiment.

| Concurrency | Method | Requests | TTFT p95 | TTFT p99 | E2E p95 | E2E p99 | TTFT SLO | E2E SLO |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | async_raw | 10 | 424.596 | 428.952 | 534.020 | 536.962 | 100.0% | 100.0% |
| 2 | async_cuszp_1e-5 | 10 | 342.748 | 343.149 | 458.710 | 460.476 | 100.0% | 100.0% |
| 4 | async_raw | 20 | 938.830 | 940.738 | 1051.754 | 1052.645 | 100.0% | 100.0% |
| 4 | async_cuszp_1e-5 | 20 | 774.062 | 776.232 | 860.089 | 863.656 | 100.0% | 100.0% |
| 8 | async_raw | 40 | 1978.170 | 1991.041 | 2088.343 | 2103.575 | 100.0% | 100.0% |
| 8 | async_cuszp_1e-5 | 40 | 1616.567 | 1621.554 | 1732.247 | 1742.481 | 100.0% | 100.0% |
| 16 | async_raw | 80 | 4067.681 | 4108.719 | 4183.787 | 4225.281 | 48.8% | 56.2% |
| 16 | async_cuszp_1e-5 | 80 | 3417.762 | 3472.807 | 3501.063 | 3559.333 | 56.2% | 68.8% |
