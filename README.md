# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is part of the Final Year Project (FYP) 2026 at **The Hong Kong Polytechnic University**. 
We integrate the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine to optimize KV Cache swapping between CPU and GPU, specifically targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

## 📁 Project Structure
```text
PolyU_COMP_Final_Year_Project_2026_Spring/
├── benchmarks/               # Performance profiling scripts
├── docker/                   # Docker infrastructure (Dockerfile, run.sh)
├── final_report/             # Final FYP thesis (PDF)
├── integration/              # Core source code (C++/Python)
├── requirements.txt          # Python dependencies
├── run.sh                    # 🚀 Root Automation Script (Builds & Tests)
└── README.md                 # This comprehensive guide
```

---

## 🛠️ Phase 1: Local Environment & Permission Setup

Before starting the Docker process, you must prepare the root directory and ensure all scripts are executable.

### 1. Grant Permissions (Mandatory)
On Linux/WSL2, scripts created in Windows often lack execution bits.
```bash
sudo chmod +x run.sh docker/run.sh
```

### 2. Create Virtual Environment (Root Directory)
Create the `venv` inside the project root to manage dependencies locally.
```bash
# 1. Ensure you are in the project root
cd PolyU_COMP_Final_Year_Project_2026_Spring

# 2. Create the environment
python3 -m venv venv

# 3. Activate the environment
source venv/bin/activate

# 4. Install requirements (Using sudo path to ensure consistency)
sudo ./venv/bin/pip install -r requirements.txt
```

---

## 🐳 Phase 2: Docker Workflow (Build then Run)

Docker handles the complex CUDA and PyTorch dependencies. Follow the order: **Build** -> **Run**.

### 1. Build the Docker Image
This step "bakes" the environment. You only need to do this once unless the `Dockerfile` or `requirements.txt` changes.
```bash
cd docker
sudo ./run.sh build
```

### 2. Run the Container
This starts the environment and mounts your code into `/workspace`.
```bash
sudo ./run.sh run
```

---

## 🚀 Phase 3: Execution inside the Container

Once you are inside the container (your terminal should show `root@container_id:/workspace#`), you run the root automation script to perform the actual work.

```bash
# You are now at the root of the workspace inside Docker
# Run the main automation script to compile C++ and run benchmarks
sudo ./run.sh
```

**What happens now?**
1.  **Automated C++ Compilation**: Compiles the cuSZp PyBind11 wrapper.
2.  **PCIe Profiling**: Measures raw H2D/D2H transfer speeds.
3.  **Compression Benchmark**: Runs the cuSZp performance and accuracy tests.

---

## ⚠️ Cross-Platform Troubleshooting (Windows/WSL2)

### 1. The "Command Not Found" Error (LF vs CRLF)
If `sudo ./run.sh` fails even after `chmod`, it's likely Windows line endings.
* **Fix**: In VS Code, check the bottom-right corner. If it says **CRLF**, click it, change to **LF**, and save. 
* **Command Fix**: `sudo sed -i 's/\r$//' run.sh`

### 2. Permanent Permission Lock (Git)
To stop Git from losing your `chmod +x` settings:
```bash
git update-index --chmod=+x run.sh
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