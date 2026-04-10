### Compression Metrics vs Baseline
| Model | Ratio | Max Error | Base Swap-Out | Eff. Swap-Out | Out Speedup | Base Swap-In | Eff. Swap-In | In Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **gpt2** | `2.69x` | `2.06e-03` | 8.25 GB/s | **14.60 GB/s** | **1.77x** | 17.16 GB/s | **32.48 GB/s** | **1.89x** |
| **Qwen_Qwen2.5-0.5B** | `2.42x` | `2.52e-02` | 14.29 GB/s | **19.53 GB/s** | **1.37x** | 17.92 GB/s | **30.52 GB/s** | **1.70x** |
| **Qwen_Qwen2.5-1.5B** | `3.06x` | `6.08e-02` | 14.56 GB/s | **22.24 GB/s** | **1.53x** | 16.32 GB/s | **33.86 GB/s** | **2.08x** |
| **facebook_opt-125m** | `2.63x` | `1.27e-03` | 15.05 GB/s | **21.61 GB/s** | **1.44x** | 20.75 GB/s | **37.36 GB/s** | **1.80x** |
| **facebook_opt-350m** | `2.68x` | `3.17e-04` | 12.33 GB/s | **18.60 GB/s** | **1.51x** | 16.52 GB/s | **31.00 GB/s** | **1.88x** |
| **EleutherAI_pythia-160m** | `2.69x` | `2.83e-03` | 13.20 GB/s | **19.77 GB/s** | **1.50x** | 16.63 GB/s | **32.25 GB/s** | **1.94x** |
| **EleutherAI_pythia-410m** | `2.79x` | `2.56e-03` | 14.41 GB/s | **21.78 GB/s** | **1.51x** | 18.38 GB/s | **35.41 GB/s** | **1.93x** |
| **TinyLlama_TinyLlama-1.1B-intermediate-step-1431k-3T** | `2.85x` | `2.04e-03` | 14.29 GB/s | **21.69 GB/s** | **1.52x** | 18.93 GB/s | **36.11 GB/s** | **1.91x** |
