# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is the Final Year Project (FYP) 2026 of the Department of Computing at **The Hong Kong Polytechnic University**. My name is **KONG Zirui** and my student ID is **22103493D**.
I integrate the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine to optimize KV Cache swapping between CPU and GPU, specifically targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

---

## 📁 Project Structure
```text
PolyU_COMP_Final_Year_Project_2026_Spring/
├── benchmarks/               # Performance profiling and experimental scripts (Python)
│   ├── benchmark_pipeline.py         # Unified benchmark pipeline for testing compression performance across multiple models
│   └── test_vllm_integration.py      # Mock CacheEngine monkey patch validation to test cuSZp integration with vLLM
├── data/                     # Output directory for `.pt` tensor caches and `.json` benchmark results
├── docker/                   # Docker infrastructure (Dockerfile, run.sh)
├── final_report/             # Final FYP thesis (PDF)
├── integration/              # Core source code (C++/Python bindings)
│   ├── compression_pipeline/         # Python hook for vLLM (compressed_swap.py)
│   └── cuszp_wrapper/                # cuSZp PyBind11 C++ Wrapper
├── interim_report/           # FYP Interim Report (PDF)
├── project_proposal/         # FYP Proposal (PDF)
├── requirements.txt          # Python dependencies
├── test.sh                   # Root Automation Script (Builds & Tests)
├── integrated_validation.py  # Unified script for benchmarks and vLLM integration tests
└── README.md                 # This project guide
```

---

## 🚀 Execution Guide

You can run this project in two ways: **via Docker** (Recommended for quick testing without polluting your host) or **Natively on Linux/WSL2** (Recommended for development and direct hardware access).

### Option A: Run via Docker (Recommended)
Docker handles the complex CUDA and PyTorch dependencies automatically. This is the safest way to avoid dependency hell.

**0. Install Prerequisites (Ubuntu/Debian)**
Ensure you have Docker and the NVIDIA Container Toolkit installed:
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

**1. Grant permissions and resolve Windows line ending issues**
```bash
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile
```

**2. Build the Docker Environment**
This step "bakes" the environment, automatically pulling PyTorch, CUDA, and compiling the original cuSZp framework into the image.
```bash
cd docker
./run.sh build
```

**3. Execute the Pipeline**
This runs `./test.sh` inside the ephemeral container and mounts your local files so results are saved to your host:
```bash
./run.sh test
```

*(Optional) Interactive debugging:*
If you need to manually debug or develop code inside the isolated environment, use `./run.sh run` to open an interactive bash session.

---

### Option B: Run Natively (Linux / WSL2)

[!CAUTION]
Warning: This method requires a perfectly configured C++ development environment. If you do not have cmake (version 3.18+), gcc/g++ (11+), and the CUDA Toolkit correctly set in your $PATH, the compilation will fail. If you encounter environment errors, please use the Docker method above.

If you are developing actively, you might want to run natively to utilize your host IDE's code completion (e.g., VS Code Pylance/C++ Intellisense) and avoid Docker overhead.

**0. Install System Dependencies**
Ensure you have `git`, `cmake`, `build-essential`, and Python dependencies installed. You must also have the **CUDA Toolkit** installed manually beforehand.
```bash
sudo apt-get update 
sudo apt-get install -y git cmake build-essential python3.10-dev python3-venv python3-pip
```

**1. Grant permissions and resolve Windows line ending issues**
```bash
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile
```

**2. Install cuSZp Core Library globally**
Our C++ Wrapper expects the `cuSZp` core framework to be located at `/opt/cuSZp`. You must compile it globally:
```bash
sudo rm -rf /opt/cuSZp
sudo git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp
cd /opt/cuSZp
sudo mkdir -p build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
sudo make -j$(nproc)
sudo make install
cd - # Return to project root
```

**3. Prepare Python Environment**
Create a local virtual environment to avoid polluting global Python packages.
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

**4. Run the Pipeline**
Simply execute the root script. It will compile the C++ PyBind11 wrapper and trigger the benchmarks:
```bash
./test.sh
```

## 📊 Benchmarks on Real KV Cache (8 Pretrained Models)

To prove our `cuSZp` KV cache swapping wrapper performs exceptionally without cherry-picking random noise data, we automatically dump and slice the true Layer-0 key embeddings directly from 8 state-of-the-art causal language models using HuggingFace. 

### Compression Metrics vs Baseline
By integrating our `compress_swap` mechanism, the end-to-end effective bandwidths (including compression overhead) are compared below against their respective raw baseline transfer speeds. With an absolute error boundary target configured at `1e-4`, it achieves the following metrics on an RTX 5080 when swapping contiguous block elements:

| Model | Compression Ratio | Absolute Max Error | Baseline Swap-Out | Effective Swap-Out | Out Speedup | Baseline Swap-In | Effective Swap-In | In Speedup |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2 (124m)** | `2.69x` | `2.06e-03` | 3.08 GB/s | **6.03 GB/s** | **1.96x** | 19.06 GB/s | **20.38 GB/s** | **1.07x** |
| **facebook/opt-125m** | `2.63x` | `1.27e-03` | 3.08 GB/s | **5.37 GB/s** | **1.74x** | 19.06 GB/s | **15.72 GB/s** | **0.82x** |
| **facebook/opt-350m** | `2.68x` | `3.17e-04` | 3.08 GB/s | **5.95 GB/s** | **1.93x** | 19.06 GB/s | **19.81 GB/s** | **1.04x** |
| **EleutherAI/pythia-160m** | `2.69x` | `2.83e-03` | 3.08 GB/s | **5.84 GB/s** | **1.90x** | 19.06 GB/s | **20.96 GB/s** | **1.10x** |
| **EleutherAI/pythia-410m** | `2.79x` | `2.56e-03` | 3.08 GB/s | **5.92 GB/s** | **1.92x** | 19.06 GB/s | **18.63 GB/s** | **0.98x** |
| **Qwen/Qwen2.5-0.5B** | `2.42x` | `2.52e-02` | 3.08 GB/s | **5.08 GB/s** | **1.65x** | 19.06 GB/s | **15.33 GB/s** | **0.80x** |
| **Qwen/Qwen2.5-1.5B** | `3.06x` | `6.08e-02` | 3.08 GB/s | **6.48 GB/s** | **2.11x** | 19.06 GB/s | **19.92 GB/s** | **1.05x** |
| **TinyLlama-1.1B** | `2.85x` | `2.04e-03` | 3.08 GB/s | **6.22 GB/s** | **2.02x** | 19.06 GB/s | **20.52 GB/s** | **1.08x** |

*Note: Once you run `./test.sh` locally, this table's baseline numbers will be automatically updated with the exact values for each specific model's cache tensor.*

*Results automatically recorded in the latest `./data` output artifacts.*

---

## 🔍 What happens during `./test.sh`?
Whether running natively or in Docker, the root automation script performs 3 main tasks:
1.  **Automated C++ Compilation**: Enters `integration/cuszp_wrapper`, cleans the build cache, and compiles the Python Wrapper for cuSZp via PyBind11 and CMake.
2.  **Unified Benchmark Pipeline**: Executes `benchmarks/benchmark_pipeline.py`. For each of the 8 models, it automatically generates real KV cache tensors, profiles the baseline PCIe H2D/D2H transfer latency, executes the cuSZp compression + transfer simulation, calculates effective speedups, and produces a single unified summary in Markdown.
3.  **vLLM Integration Simulation**: Executes `benchmarks/test_vllm_integration.py` to mock a `vllm.worker.cache_engine.CacheEngine` instance and demonstrates our `CompressedCacheEngineMonkeyPatch` correctly intercepts, compresses (`swap_out`), and recovers (`swap_in`) KV blocks transparently.

*Results will be automatically saved in `.json` files in the root directory.*

---

## ⚠️ Cross-Platform Troubleshooting (Windows/WSL2)

### 1. The "Command Not Found" Error (LF vs CRLF)
If `./test.sh` or `./run.sh` fails with strange syntax errors, it is likely due to Windows line endings (CRLF) being cloned into WSL.
* **Quick Fix**: Run `sed -i 's/\r$//' test.sh docker/run.sh`
* **VS Code Fix**: Check the bottom-right corner of the editor. If it says **CRLF**, click it, change to **LF**, and save the file.

### 2. Permanent Permission Lock (Git)
To stop Git from losing your `chmod +x` executable settings between Windows and WSL:
```bash
git update-index --chmod=+x test.sh
git update-index --chmod=+x docker/run.sh
git commit -m "chore: lock executable permissions"
```

### 3. GPU Passthrough Issues
Ensure `nvidia-smi` works on your host. If Docker can't see the GPU, you must re-check the **NVIDIA Container Toolkit** installation on your WSL2/Linux host.

---

## 📑 Project Reports
* **Final Report**: [final_report/FYP_2026_Final_Report.pdf](final_report/FYP_2026_Final_Report.pdf)
* **Interim Report**: [interim_report/FYP_2026_Interim_Report.pdf](interim_report/FYP_2026_Interim_Report.pdf)
* **Initial Proposal**: [project_proposal/Compress_Transfer_Decompress.pdf](project_proposal/Compress_Transfer_Decompress.pdf)
