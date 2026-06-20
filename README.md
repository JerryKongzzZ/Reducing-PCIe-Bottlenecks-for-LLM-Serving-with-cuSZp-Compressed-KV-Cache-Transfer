# Artifact Evaluation: Congestion-Aware KV Cache Swapping for High-Throughput LLM Serving

This repository contains the source code, microbenchmarks, and artifact evaluation scripts for our system submitted to top-tier networking/systems conferences (e.g., INFOCOM).

Our system addresses the **CPU-GPU PCIe Memory Wall** in LLM serving. By modeling the PCIe bus as a bandwidth-constrained network link and KV Cache swapping as traffic flows with varying QoS requirements, we introduce a **Congestion-Aware Adaptive Lossy Compression Scheduler**. Under high-concurrency (burst) traffic, it automatically sacrifices precision in less-sensitive deep network layers to instantly clear the PCIe queue, preventing congestion storms and achieving Pareto-optimal latency and throughput.

---

## 🏗️ System Architecture

Our system consists of three decoupled modules:
1. **Bottom-Level Engine (`integration/cuszp_wrapper/`)**: A PyBind11 C++ wrapper over `cuSZp` that intercepts PyTorch's native CUDA streams to perform non-blocking concurrent compression without CPU syncs.
2. **vLLM Hijack Layer (`integration/compression_pipeline/compressed_swap.py`)**: A zero-intrusion monkey-patch for vLLM's `CacheEngine`, dynamically intercepting `_swap_in` and `_swap_out` to introduce independent metadata stores and prioritized asynchronous decompression.
3. **The Brain (`integration/compression_pipeline/adaptive_scheduler.py`)**: The PCIe congestion-aware scheduler. It calculates pending bytes in the PCIe queue, categorizes congestion into `GREEN/YELLOW/RED` states, and greedily allocates relative error bounds (`eps`) based on the transformer layer's offline-profiled sensitivity.

---

## 🛠️ Hardware & Software Requirements

- **OS**: Linux (Ubuntu 20.04/22.04) or WSL2
- **GPU**: NVIDIA GPU (Tested on RTX 5080) with CUDA 12.x support
- **CPU-GPU Interconnect**: PCIe Gen 4.0 or 5.0
- **Software**: Python 3.10+, CMake 3.18+, GCC 11+

---

## 🚀 Getting Started (Environment Setup)

We provide instructions to set up the environment natively on Linux or WSL2. Ensure you have the CUDA Toolkit installed before proceeding.

```bash
# 1. Install system dependencies
sudo apt-get update && sudo apt-get install -y git cmake build-essential python3.10-dev python3-venv

# 2. Compile and install cuSZp core library globally
sudo git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp
cd /opt/cuSZp && sudo mkdir -p build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
sudo make -j$(nproc) && sudo make install
cd -

# 3. Setup Python virtual environment
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

---

## 🧪 Detailed Evaluation Instructions

This section guides you through reproducing the core claims of our paper.

### Claim 1: cuSZp Compression dramatically improves Effective PCIe Bandwidth
We extract actual Layer-0 KV cache embeddings from 8 state-of-the-art models via HuggingFace and benchmark the physical PCIe transfer time vs. our compressed transfer time.

**How to reproduce:**
```bash
./test.sh
```
**Expected Outcome:** 
The script will compile the C++ PyBind11 wrapper and profile the baseline PCIe transfer latency against the cuSZp compression + transfer simulation. A summary table will be printed to the console, and detailed `.json` logs will be saved in the `data/` directory. You should observe ~1.5x - 2.0x effective bandwidth speedups across all 8 models.

### Claim 2: Deep Transformer Layers are robust to high-ratio Lossy Compression
We profile the KL-divergence of compressing individual Transformer layers to build an offline sensitivity map.

**How to reproduce:**
```bash
# Generate the layer sensitivity map for GPT-2
PYTHONPATH=integration/compression_pipeline python3 benchmarks/layer_sensitivity_sweep.py \
    --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2
```
**Expected Outcome:** 
A `layer_sensitivity.json` file is generated in the `data/` folder, classifying layers into `shallow`, `mid`, and `deep` categories based on their tolerance to compression errors.

### Claim 3: Adaptive Congestion-Aware Scheduling achieves Pareto-Optimality
We evaluate the full adaptive system against baselines (Uncompressed, Static cuSZp, INT8, ZLIB) under varying PCIe queue depths.

**How to reproduce:**
```bash
# Run the end-to-end policy evaluation
PYTHONPATH=integration/compression_pipeline python3 benchmarks/evaluate_policies.py \
    --models gpt2 --out data/eval_summary.json

# Generate the paper figures (Pareto Boundary, Queue Waterfall, Ablation)
python3 benchmarks/run_pareto_queue_ablation.py
```
**Expected Outcome:** 
The script generates three plots in the `data/figures/` directory:
1. `pareto_boundary.png`: Shows our adaptive scheduler maintaining high accuracy while achieving maximum throughput.
2. `queue_depth_waterfall.png`: Demonstrates how the `RED` congestion state instantly drains pending PCIe volume during burst traffic.
3. `ablation_study.png`: Highlights the latency (TTFT) reduction contributed by our asynchronous decompression pipeline.

---

## 💻 API Usage: Enabling the Scheduler in vLLM

To integrate our system into an existing vLLM deployment, simply apply our zero-intrusion monkey-patch before engine execution:

```python
from integration.compression_pipeline.compressed_swap import setup_vllm_compression
import vllm

# Initialize your vLLM engine
engine = vllm.LLMEngine(...)

# Apply the Congestion-Aware Monkey Patch
# enable_adaptive=True enables dynamic queue monitoring and GREEN/YELLOW/RED state shifts
patcher = setup_vllm_compression(engine, error_bound=1e-4, enable_adaptive=True)

# Run your inference workload...

# To safely remove the patch:
patcher.unpatch()
```
<<<<<<< Updated upstream

---

## 📊 Microbenchmark Results (Hardware: RTX 5080)

To prove our `cuSZp` KV cache swapping wrapper performs exceptionally without cherry-picking random noise data, we automatically dump and slice the true Layer-0 key embeddings directly from 8 state-of-the-art causal language models using HuggingFace. 

By integrating our `compress_swap` mechanism, the end-to-end effective bandwidths (including compression overhead) are compared below against their respective raw baseline transfer speeds. With an absolute error boundary target configured at `1e-4`:

| Model | Compression Ratio | Absolute Max Error | Baseline Swap-Out | Effective Swap-Out | Out Speedup | Baseline Swap-In | Effective Swap-In | In Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2 (124m)** | `2.69x` | `2.06e-03` | 8.25 GB/s | **14.60 GB/s** | **1.77x** | 17.16 GB/s | **32.48 GB/s** | **1.89x** |
| **Qwen_Qwen2.5-0.5B** | `2.42x` | `2.52e-02` | 14.29 GB/s | **19.53 GB/s** | **1.37x** | 17.92 GB/s | **30.52 GB/s** | **1.70x** |
| **Qwen_Qwen2.5-1.5B** | `3.06x` | `6.08e-02` | 14.56 GB/s | **22.24 GB/s** | **1.53x** | 16.32 GB/s | **33.86 GB/s** | **2.08x** |
| **facebook_opt-125m** | `2.63x` | `1.27e-03` | 15.05 GB/s | **21.61 GB/s** | **1.44x** | 20.75 GB/s | **37.36 GB/s** | **1.80x** |
| **facebook_opt-350m** | `2.68x` | `3.17e-04` | 12.33 GB/s | **18.60 GB/s** | **1.51x** | 16.52 GB/s | **31.00 GB/s** | **1.88x** |
| **EleutherAI_pythia-160m** | `2.69x` | `2.83e-03` | 13.20 GB/s | **19.77 GB/s** | **1.50x** | 16.63 GB/s | **32.25 GB/s** | **1.94x** |
| **EleutherAI_pythia-410m** | `2.79x` | `2.56e-03` | 14.41 GB/s | **21.78 GB/s** | **1.51x** | 18.38 GB/s | **35.41 GB/s** | **1.93x** |
| **TinyLlama-1.1B** | `2.85x` | `2.04e-03` | 14.29 GB/s | **21.69 GB/s** | **1.52x** | 18.93 GB/s | **36.11 GB/s** | **1.91x** |

*Note: Baseline numbers are automatically updated dynamically upon running `./test.sh` locally.*

---

## ⚠️ Troubleshooting (Windows/WSL2)

### 1. The "Command Not Found" Error (LF vs CRLF)
If shell scripts fail with syntax errors, it is likely due to Windows line endings (CRLF).
* **Quick Fix**: Run `sed -i 's/\r$//' test.sh docker/run.sh`

### 2. Permanent Permission Lock (Git)
To stop Git from losing your executable settings between Windows and WSL:
```bash
git update-index --chmod=+x test.sh
git update-index --chmod=+x docker/run.sh
git commit -m "chore: lock executable permissions"
```
=======
>>>>>>> Stashed changes
