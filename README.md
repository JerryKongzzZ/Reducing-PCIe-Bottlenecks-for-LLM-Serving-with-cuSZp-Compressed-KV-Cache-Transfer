# Memory Wall is a Network Problem: Congestion-Aware KV Cache Swapping for High-Throughput LLM Serving

## 🌟 Project Overview
This project proposes a novel system design targeting the **CPU-GPU PCIe Memory Wall** during large language model (LLM) serving. By modeling the PCIe bus as a bandwidth-constrained network link and KV Cache swapping as traffic flows with varying Quality-of-Service (QoS) requirements, we introduce a **Congestion-Aware Adaptive Lossy Compression Scheduler**.

This system dynamically balances PCIe communication overhead and model inference accuracy. Under high-concurrency (burst) traffic, it automatically sacrifices precision in less-sensitive deep network layers to instantly clear the PCIe queue, effectively preventing congestion storms and achieving Pareto-optimal end-to-end latency and throughput. 

This repository contains the full source code, benchmark suites, and plotting scripts intended for **Artifact Evaluation at top-tier networking and systems conferences (e.g., INFOCOM, OSDI, SIGCOMM)**. It seamlessly integrates the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine, targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

### 🏗️ System Architecture
Our system consists of three decoupled modules:
1. **Bottom-Level Engine (`integration/cuszp_wrapper/`)**: A PyBind11 C++ wrapper over `cuSZp` that intercepts PyTorch's native CUDA streams to perform non-blocking concurrent compression without CPU syncs.
2. **vLLM Hijack Layer (`integration/compression_pipeline/compressed_swap.py`)**: A zero-intrusion monkey-patch for vLLM's `CacheEngine`, dynamically intercepting `_swap_in` and `_swap_out` to introduce independent metadata stores and prioritized asynchronous decompression.
3. **The Brain (`integration/compression_pipeline/adaptive_scheduler.py`)**: The PCIe congestion-aware scheduler. It calculates pending bytes in the PCIe queue, categorizes congestion into `GREEN/YELLOW/RED` states, and greedily allocates relative error bounds (`eps`) based on the transformer layer's offline-profiled sensitivity.

---

## 📁 Project Structure
```text
PolyU_COMP_Final_Year_Project_2026_Spring/
├── benchmarks/               # Experimental scripts and artifact evaluation
│   ├── benchmark_pipeline.py         # Microbenchmarks across 8 models
│   ├── layer_sensitivity_sweep.py    # Offline profiling for layer sensitivity
│   ├── evaluate_policies.py          # End-to-end policy evaluation (Adaptive vs Static)
│   ├── run_pareto_queue_ablation.py  # Generates paper-ready figures (Pareto, Queue, Ablation)
│   └── test_vllm_integration.py      # Monkey-patch validation
├── data/                     # Output directory for `.json` results and `figures/`
├── docker/                   # Docker infrastructure (Dockerfile, run.sh)
├── integration/              # Core source code (C++/Python bindings)
│   ├── compression_pipeline/         # Python hook & Adaptive Scheduler for vLLM
│   └── cuszp_wrapper/                # cuSZp PyBind11 C++ Wrapper
├── documents/                # Legacy FYP documents (Proposal, Interim, Final Report)
├── requirements.txt          # Python dependencies
├── test.sh                   # Root Automation Script (Builds & Microbenchmarks)
└── README.md                 # Artifact Evaluation Guide
```

---

## 🎯 Artifact Evaluation & Execution Guide

To reproduce the experimental results presented in the paper, we provide an automated pipeline. You can run this project in two ways: **via Docker** (Recommended for pure evaluation) or **Natively on Linux/WSL2** (Recommended for development).

### 1. Environment Setup

**Option A: Run via Docker (Recommended)**
Docker handles the complex CUDA and PyTorch dependencies automatically.
```bash
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile
cd docker
./run.sh build  # Builds the image with cuSZp and dependencies
```

**Option B: Run Natively (Linux / WSL2)**
*Requires `cmake` (3.18+), `gcc/g++` (11+), and the CUDA Toolkit.*
```bash
# 1. Install cuSZp Core Library globally
sudo git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp
cd /opt/cuSZp && sudo mkdir -p build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
sudo make -j$(nproc) && sudo make install

# 2. Setup Python environment
cd /path/to/project
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Reproducing Paper Experiments

We provide push-button scripts to generate all tables and figures used in the paper.

**Step 1: Microbenchmarks (Table Generation)**
This script automatically compiles the C++ PyBind11 wrapper, generates real KV cache tensors for 8 state-of-the-art models, profiles the baseline PCIe transfer latency, executes the cuSZp compression + transfer simulation, and calculates effective speedups.
```bash
# If using Docker:
./docker/run.sh test
# If running natively:
./test.sh
```
*Output: Detailed bandwidth/latency json files saved in `data/`.*

**Step 2: Generate Offline Layer Sensitivity Profile**
Profiles the KL-divergence of compressing individual Transformer layers to build the sensitivity map.
```bash
PYTHONPATH=integration/compression_pipeline python3 benchmarks/layer_sensitivity_sweep.py --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2
```

**Step 3: End-to-End Evaluation & Generating Paper Figures**
Evaluates the full adaptive system against baselines (Static cuSZp, Uncompressed vLLM) and plots the Pareto boundaries, queue depths, and ablation studies.
```bash
PYTHONPATH=integration/compression_pipeline python3 benchmarks/evaluate_policies.py --models gpt2 --out data/eval_summary.json

# Generate INFOCOM/Top-Tier conference specific evaluation plots (Pareto, Queue Depth, Ablation)
python3 benchmarks/run_pareto_queue_ablation.py
```
*Output: `pareto_boundary.png`, `queue_depth_waterfall.png`, and `ablation_study.png` will be saved in `data/figures/`.*

### 3. API Usage: Enabling the Scheduler in vLLM
Our integration exposes a zero-intrusion monkey-patch for vLLM. It can be initialized in a few lines of code:
```python
from integration.compression_pipeline.compressed_swap import setup_vllm_compression

# Inside your vLLM engine initialization code:
# enable_adaptive=True turns on automatic per-layer sensitivity inference and PCIe queue monitoring
patcher = setup_vllm_compression(engine, error_bound=1e-4, enable_adaptive=True)
```

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

---

## 🎓 Legacy Project Documentation (FYP 2026)
This project originated as a Capstone Project (FYP 2026) at The Hong Kong Polytechnic University by KONG Zirui (22103493D).
* **Final Report**: [documents/final_report/Capstone_Project_2026_Final_Report_KONG_Zirui_22103493D.pdf](documents/final_report/Capstone_Project_2026_Final_Report_KONG_Zirui_22103493D.pdf)
* **Interim Report**: [documents/interim_report/FYP_2026_Interim_Report.pdf](documents/interim_report/FYP_2026_Interim_Report.pdf)
* **Initial Proposal**: [documents/project_proposal/Compress_Transfer_Decompress_for_LLM_Serving__cuSZp_Enabled_CPU_GPU_Data_Pipeline_in_vLLM.pdf](documents/project_proposal/Compress_Transfer_Decompress_for_LLM_Serving__cuSZp_Enabled_CPU_GPU_Data_Pipeline_in_vLLM.pdf)
