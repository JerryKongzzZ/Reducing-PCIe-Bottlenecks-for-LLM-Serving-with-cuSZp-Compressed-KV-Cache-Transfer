# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is the Final Year Project (FYP) 2026 of the Department of Computing at **The Hong Kong Polytechnic University**. My name is **KONG Zirui** and my student ID is **22103493D**.
I integrate the **cuSZp** error-bounded lossy compression framework into the **vLLM** engine to optimize KV Cache swapping between CPU and GPU, specifically targeting performance gains on high-bandwidth hardware like the **RTX 5080**.

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

---

## 🔍 What happens during `./test.sh`?
Whether running natively or in Docker, the root automation script performs 5 main tasks:
1.  **Automated C++ Compilation**: Enters `integration/cuszp_wrapper`, cleans the build cache, and compiles the Python Wrapper for cuSZp via PyBind11 and CMake.
2.  **KV Cache Dataset Generation**: Extracts real Layer-0 KV Cache parameters from a HuggingFace causal LM (`gpt2`) to simulate the true distribution of token states in production models.
3.  **PCIe Profiling**: Executes `benchmarks/baseline_profiling.py` to measure raw H2D/D2H tensor transfer latency under PCIe 4.0/5.0.
4.  **Compression Benchmark**: Executes `benchmarks/compression_benchmark.py` over the *real KV Cache*, dynamically calculating relative error boundaries based on exact tensor ranges (Min-Max Extraction). Verifies compression ratio (~2.5x), throughput (~9 GB/s), and absolute max precision error (bounded precisely at ~1e-3).
5.  **vLLM Integration Simulation**: Executes `benchmarks/test_vllm_integration.py` to mock a `vllm.worker.cache_engine.CacheEngine` instance and demonstrates our `CompressedCacheEngineMonkeyPatch` correctly intercepts, compresses (`swap_out`), and recovers (`swap_in`) KV blocks transparently.

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
