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
