# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is part of the Final Year Project (FYP) 2026 at **The Hong Kong Polytechnic University**. 
We integrate the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine to optimize KV Cache swapping between CPU and GPU, specifically targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

## 🚀 TL;DR: Quick Start Options
You can run this project in two ways: **via Docker** (Recommended for quick testing without polluting your host) or **Natively on Linux/WSL2** (Recommended for development and direct hardware access).

### Option A: Run via Docker (Easiest)
*Prerequisites: Docker and NVIDIA Container Toolkit must be installed.*
```bash
# 1. Grant execution permissions and resolve Windows line ending issues
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile

# 2. Enter the Docker directory, build the environment, and run the pipeline
cd docker
./run.sh build
./run.sh test
```
*Results will be automatically saved in `.json` files in the root directory.*

### Option B: Run Natively (Linux / WSL2)
*Prerequisites: Python 3.10+, CMake, and CUDA Toolkit must be installed on your host system.*
```bash
# 1. Grant execution permissions and resolve Windows line ending issues
chmod +x test.sh docker/run.sh
sed -i 's/\r$//' test.sh docker/run.sh docker/Dockerfile

# 2. Install cuSZp core library globally (Required by CMakeLists)
sudo rm -rf /opt/cuSZp
sudo git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp
cd /opt/cuSZp && sudo mkdir -p build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
sudo make -j$(nproc) && sudo make install
cd - # Return to project root

# 3. Setup Python virtual environment and install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 4. Run the automated test script
./test.sh
```

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

## 🐳 Detailed Guide A: Docker Workflow (Recommended)

Docker handles the complex CUDA and PyTorch dependencies automatically. This is the safest way to avoid dependency hell.

### 1. Build the Docker Image
This step "bakes" the environment, automatically pulling PyTorch, CUDA, and compiling the original cuSZp framework into the image.
```bash
cd docker
./run.sh build
```

### 2. Execute the Pipeline
This runs `./test.sh` inside the ephemeral container and mounts your local files so results are saved to your host:
```bash
cd docker
./run.sh test
```

### 3. Enter Interactive Container (Optional)
If you need to manually debug or develop code inside the isolated environment:
```bash
cd docker
./run.sh run
# Once inside the container (root@container_id:/workspace#), you can manually run:
# ./test.sh
```

---

## 💻 Detailed Guide B: Native Linux / WSL2 Workflow

If you are developing actively, you might want to run natively to utilize your host IDE's code completion (e.g., VS Code Pylance/C++ Intellisense) and avoid Docker overhead.

### 1. Install System Dependencies
Ensure you have `git`, `cmake`, `build-essential`, and a valid `CUDA Toolkit` (nvcc) installed.
```bash
sudo apt-get update && sudo apt-get install -y git cmake build-essential python3.10-dev python3-pip
```

### 2. Install cuSZp Core Library
Our C++ Wrapper expects the `cuSZp` core framework to be located at `/opt/cuSZp`. You must compile it globally:
```bash
sudo rm -rf /opt/cuSZp
sudo git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp
cd /opt/cuSZp
sudo mkdir -p build && cd build
sudo cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ ..
sudo make -j$(nproc)
sudo make install
```

### 3. Prepare Python Environment
Create a local virtual environment to avoid polluting global Python packages.
```bash
cd /path/to/PolyU_COMP_Final_Year_Project_2026_Spring
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Run the Pipeline
Simply execute the root script. It will compile the C++ PyBind11 wrapper and trigger the benchmarks:
```bash
./test.sh
```

---

## 🔍 What happens during `./test.sh`?
Whether running natively or in Docker, the root automation script performs 3 main tasks:
1.  **Automated C++ Compilation**: Enters `integration/cuszp_wrapper`, cleans the build cache, and compiles the Python Wrapper for cuSZp via PyBind11 and CMake.
2.  **PCIe Profiling**: Executes `benchmarks/baseline_profiling.py` to measure raw H2D/D2H tensor transfer latency under PCIe 4.0/5.0.
3.  **Compression Benchmark**: Executes `benchmarks/compression_benchmark.py` to run the cuSZp compression performance tests, verifying compression ratio, throughput, and maximum absolute error (converging to 1e-4).

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
