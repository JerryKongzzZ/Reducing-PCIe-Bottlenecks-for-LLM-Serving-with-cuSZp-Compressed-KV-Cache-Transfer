# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview

This project is a Final Year Project (FYP) for the Spring 2026 semester at The Hong Kong Polytechnic University. 
It aims to deeply integrate the ultra-fast GPU compression library **cuSZp** into the **vLLM** large language model inference framework. By introducing an innovative "compress-transfer-decompress" workflow, it optimizes CPU-GPU KV Cache data exchange (Swap In/Out), significantly reducing the negative impact of PCIe bandwidth bottlenecks on LLM throughput and latency.

## 📁 Project Structure

```text
FYP_Workspace/
├── cuSZp/                    # Official cuSZp compression library (Git Submodule/Clone)
├── integration/              # Core integration code
│   ├── cuszp_wrapper/        # cuSZp C++ and PyBind11 wrappers
│   └── compression_pipeline/ # Python-level compression pipeline logic
├── benchmarks/               # Performance benchmarking scripts
├── docker/                   # Docker configurations & usage guide (contains its own run.sh)
├── run.sh                    # 🚀 Automated one-click build and test script (Root)
└── README.md                 # Project overview documentation
```

## 🚀 Quick Start (One-Click Execution)

This project is highly engineered and provides a fully automated containerized testing environment. It is strongly recommended that all users (both Windows and Linux) use Docker for compilation and testing.

> ⚠️ **Prerequisites**: Your host machine must have an NVIDIA GPU, and Docker along with the NVIDIA Container Toolkit must be properly installed. If you are a Windows (WSL2) user, please read [`docker/README.md`](docker/README.md) first to complete your environment setup.

### 3 Steps to Run the Core Benchmarks:

**1. Start and enter the clean development container**
Use the infrastructure script located in the `docker` folder to start the environment:
```bash
cd docker
./run.sh run
```

**2. Execute the automated test script (Root Directory)**
Once inside the container (your working directory will be `/workspace`), directly run the root-level automated script. 
*(Note: Be careful not to confuse this root `run.sh` with the `docker/run.sh` used in step 1).*
```bash
./run.sh
```
This script will automatically clean the environment, compile the C++ PyBind11 extension, and sequentially execute the baseline bandwidth profiling and cuSZp compression benchmarks.

**3. View the performance report**
After the tests are completed, core metrics such as throughput (GB/s), compression ratio, and error bounds will be printed in the terminal and automatically saved as JSON report files (`baseline_results.json` & `compression_results.json`).

## 📈 Project Implementation Status

- [x] **Phase 1: Architecture Design** - Analyzed vLLM source code structure and CPU-GPU data transfer bottlenecks.
- [x] **Phase 2: Core Bridging** - Developed a C++ Wrapper for the cuSZp API and exposed it to Python via PyBind11.
- [x] **Phase 3: Pipeline Integration** - Resolved CMake linking, dynamic enum types, and Docker mount masking issues, successfully compiling the shared library (`.so`).
- [x] **Phase 4: Performance Validation** - Completed PCIe baseline bandwidth analysis and cuSZp compression throughput evaluation (Achieved nearly 8 GB/s one-way throughput on RTX 5080).
- [ ] **Phase 5: vLLM Integration** - Pending the integration of `CompressedSwapManager` into vLLM's `CacheEngine` for real-world end-to-end LLM inference testing.

## 📚 References

1. Huang, Y., et al. (2023). cuSZp: An Ultra-fast GPU Error-bounded Lossy Compression Framework. SC'23.
2. Kwon, W., et al. (2023). Efficient memory management for large language model serving with pagedattention. SOSP'23.