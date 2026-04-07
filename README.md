# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is part of the Final Year Project (FYP) 2026 at **The Hong Kong Polytechnic University**. 
We integrate the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine to optimize KV Cache swapping between CPU and GPU, specifically targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

## 🚀 TL;DR: One-Click Quick Start
If you just want to run all the compilation and performance benchmark tests, simply copy the following script and execute it once in your host terminal (assuming Docker and NVIDIA Container Toolkit are installed):

```bash
# 1. Grant execution permissions and resolve Windows line ending issues (Safe for Mac/Linux users as well)
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile

# 2. Enter the Docker directory and automatically build the environment
cd docker
./run.sh build

# 3. Automatically mount the directory and execute the test pipeline (C++ Compilation + Baseline + cuSZp Tests)
./run.sh test
```
*After execution, the performance test results of all Benchmarks will be automatically saved in `.json` files generated in the root directory.*

---

## 📁 Project Structure
```text
PolyU_COMP_Final_Year_Project_2026_Spring/
├── benchmarks/               # Performance profiling scripts (Python)
├── docker/                   # Docker infrastructure (Dockerfile, run.sh)
├── final_report/             # Final FYP thesis (PDF)
├── integration/              # Core source code (C++/Python bindings)
├── requirements.txt          # Python dependencies
├── test.sh                   # 🚀 Root Automation Script (Builds & Tests)
└── README.md                 # This comprehensive guide
```

---

## 🛠️ Detailed Guide: Phase 1 (Local Environment Setup)

### 1. Grant Permissions (Mandatory)
On Linux/WSL2, scripts created in Windows often lack execution bits.
```bash
chmod +x test.sh docker/run.sh
```

### 2. Create Virtual Environment (Optional, for IDE Auto-completion)
If you want better code completion experience in VS Code on your host machine, you can create a venv in the project root:
```bash
# 1. Ensure you are in the project root
cd PolyU_COMP_Final_Year_Project_2026_Spring

# 2. Create the environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install requirements locally
./venv/bin/pip install -r requirements.txt
```

---

## 🐳 Detailed Guide: Phase 2 (Docker Workflow)

Docker handles the complex CUDA and PyTorch dependencies. Follow the order: **Build** -> **Run**.

### 1. Build the Docker Image
This step "bakes" the environment, automatically pulling PyTorch, CUDA, and compiling the original cuSZp framework.
```bash
cd docker
./run.sh build
```

### 2. Enter Interactive Container (Optional)
If you need to manually debug or develop code, you can start an interactive shell mounting your local code:
```bash
./run.sh run
```

---

## ⚙️ Detailed Guide: Phase 3 (Execution inside Docker)

When you are inside the container (`root@container_id:/workspace#`), you can run the full pipeline through the automated test script:

```bash
# Run C++ extension compilation and automated Benchmark tests
./test.sh
```

**What happens now?**
1.  **Automated C++ Compilation**: Automatically enters the `integration/cuszp_wrapper` directory and compiles the Python Wrapper for cuSZp via PyBind11 and CMake.
2.  **PCIe Profiling**: Executes `benchmarks/baseline_profiling.py` to measure raw H2D/D2H tensor transfer latency under PCIe 4.0/5.0.
3.  **Compression Benchmark**: Executes `benchmarks/compression_benchmark.py` to run the cuSZp compression performance tests, verifying compression ratio, throughput, and maximum absolute error (converging to 1e-4).

---

## ⚠️ Cross-Platform Troubleshooting (Windows/WSL2)

### 1. The "Command Not Found" Error (LF vs CRLF)
If `./test.sh` fails even after `chmod`, it's likely Windows line endings.
* **Fix**: In VS Code, check the bottom-right corner. If it says **CRLF**, click it, change to **LF**, and save. 
* **Command Fix**: `sed -i 's/\r$//' test.sh docker/run.sh`

### 2. Permanent Permission Lock (Git)
To stop Git from losing your `chmod +x` settings:
```bash
git update-index --chmod=+x test.sh
git update-index --chmod=+x docker/run.sh
git commit -m "chore: lock executable permissions"
```

### 3. GPU Passthrough
Ensure `nvidia-smi` works on your host. If Docker can't see the GPU, you must re-check the **NVIDIA Container Toolkit** installation on your WSL2/Linux host.

---

## 📑 Project Reports
* **Final Report**: [final_report/FYP_2026_Final_Report.pdf](final_report/FYP_2026_Final_Report.pdf)
* **Interim Report**: [interim_report/FYP_2026_Interim_Report.pdf](interim_report/FYP_2026_Interim_Report.pdf)
* **Initial Proposal**: [project_proposal/Compress_Transfer_Decompress.pdf](project_proposal/Compress_Transfer_Decompress.pdf)
