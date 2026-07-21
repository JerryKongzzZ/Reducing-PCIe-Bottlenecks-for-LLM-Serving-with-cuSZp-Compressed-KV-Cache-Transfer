# Gate D multi-model summary

All methods use the asynchronous offload worker, batched restore, and
the same per-model prompt/KV-cache workload. `quality` is evaluated
per trial relative to `async_raw`.

| Model | Method | Ratio | G2C ms | C2G ms | Token | Exact | Quality |
|---|---|---:|---:|---:|---:|---:|:---:|
| gpt2 | async_raw | 1.00000x | 17.152 | 1.821 | 0.96875 | 0.87500 | pass |
| gpt2 | async_cuszp_1e-5 | 1.19784x | 18.039 | 0.454 | 0.96875 | 0.87500 | pass |
| gpt2 | async_int8 | 1.99997x | 46.790 | 2.056 | 1.00000 | 1.00000 | pass |
| gpt2 | async_zstd | 1.25141x | 64.669 | 22.015 | 0.96875 | 0.87500 | pass |
| gpt2 | async_lz4 | 1.00000x | 46.047 | 0.902 | 0.96875 | 0.87500 | pass |
| facebook/opt-125m | async_raw | 1.00000x | 37.146 | 0.993 | 0.90625 | 0.87500 | pass |
| facebook/opt-125m | async_cuszp_1e-5 | 1.15163x | 35.879 | 1.030 | 0.90625 | 0.87500 | pass |
| facebook/opt-125m | async_int8 | 1.99997x | 67.186 | 3.320 | 0.82812 | 0.75000 | reject |
| facebook/opt-125m | async_zstd | 1.22877x | 119.947 | 61.357 | 1.00000 | 1.00000 | pass |
| facebook/opt-125m | async_lz4 | 1.00000x | 94.596 | 0.995 | 1.00000 | 1.00000 | pass |
| facebook/opt-350m | async_raw | 1.00000x | 82.317 | 0.802 | 1.00000 | 1.00000 | pass |
| facebook/opt-350m | async_cuszp_1e-5 | 1.14864x | 76.851 | 0.889 | 1.00000 | 1.00000 | pass |
| facebook/opt-350m | async_int8 | 1.99999x | 105.568 | 2.715 | 0.87500 | 0.87500 | reject |
| facebook/opt-350m | async_zstd | 1.24847x | 256.346 | 104.436 | 1.00000 | 1.00000 | pass |
| facebook/opt-350m | async_lz4 | 1.00000x | 170.007 | 0.846 | 1.00000 | 1.00000 | pass |
| EleutherAI/pythia-160m | async_raw | 1.00000x | 34.894 | 1.020 | 0.56250 | 0.50000 | pass |
| EleutherAI/pythia-160m | async_cuszp_1e-5 | 1.27078x | 34.149 | 0.942 | 0.70312 | 0.62500 | pass |
| EleutherAI/pythia-160m | async_int8 | 1.99997x | 68.704 | 3.078 | 0.51562 | 0.50000 | reject |
| EleutherAI/pythia-160m | async_zstd | 1.28381x | 139.359 | 63.522 | 0.59375 | 0.50000 | pass |
| EleutherAI/pythia-160m | async_lz4 | 1.02997x | 105.301 | 36.745 | 0.59375 | 0.50000 | pass |
| EleutherAI/pythia-410m | async_raw | 1.00000x | 86.493 | 0.823 | 0.81250 | 0.62500 | pass |
| EleutherAI/pythia-410m | async_cuszp_1e-5 | 1.21876x | 67.793 | 1.020 | 0.79688 | 0.75000 | reject |
| EleutherAI/pythia-410m | async_int8 | 1.99999x | 110.262 | 2.790 | 0.56250 | 0.50000 | reject |
| EleutherAI/pythia-410m | async_zstd | 1.26487x | 291.704 | 108.972 | 0.81250 | 0.62500 | pass |
| EleutherAI/pythia-410m | async_lz4 | 1.00205x | 191.959 | 29.504 | 0.81250 | 0.62500 | pass |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | async_raw | 1.00000x | 44.961 | 0.953 | 1.00000 | 1.00000 | pass |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | async_cuszp_1e-5 | 1.24392x | 42.852 | 5.444 | 1.00000 | 1.00000 | pass |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | async_int8 | 1.99996x | 95.097 | 3.087 | 0.56250 | 0.37500 | reject |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | async_zstd | 1.21501x | 112.413 | 49.122 | 1.00000 | 1.00000 | pass |
| TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T | async_lz4 | 1.00004x | 109.095 | 1.928 | 1.00000 | 1.00000 | pass |
| Qwen/Qwen2.5-0.5B | async_raw | 1.00000x | 36.623 | 3.774 | 0.98125 | 0.97500 | pass |
| Qwen/Qwen2.5-0.5B | async_cuszp_1e-5 | 1.48128x | 42.545 | 5.324 | 1.00000 | 1.00000 | pass |
| Qwen/Qwen2.5-0.5B | async_int8 | 1.99992x | 100.197 | 6.573 | 0.60938 | 0.50000 | reject |
| Qwen/Qwen2.5-0.5B | async_zstd | 1.25044x | 134.706 | 78.593 | 1.00000 | 1.00000 | pass |
| Qwen/Qwen2.5-0.5B | async_lz4 | 1.00244x | 129.040 | 60.688 | 1.00000 | 1.00000 | pass |
| Qwen/Qwen2.5-1.5B | async_raw | 1.00000x | 69.931 | 1.895 | 0.87500 | 0.87500 | pass |
| Qwen/Qwen2.5-1.5B | async_cuszp_1e-5 | 1.61445x | 42.213 | 1.860 | 0.87500 | 0.87500 | pass |
| Qwen/Qwen2.5-1.5B | async_int8 | 1.99997x | 151.181 | 4.842 | 0.37500 | 0.37500 | reject |
| Qwen/Qwen2.5-1.5B | async_zstd | 1.26785x | 227.369 | 92.371 | 0.87500 | 0.87500 | pass |
| Qwen/Qwen2.5-1.5B | async_lz4 | 1.00101x | 193.646 | 47.413 | 0.87500 | 0.87500 | pass |
