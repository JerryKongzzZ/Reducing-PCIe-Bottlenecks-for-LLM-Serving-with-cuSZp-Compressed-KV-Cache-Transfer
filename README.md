# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is part of the Final Year Project (FYP) 2026 at **The Hong Kong Polytechnic University**. 
We integrate the **cuSZp** error-bounded lossy compression framework into **vLLM** to optimize KV Cache swapping between CPU and GPU. By compressing data before transferring it over the PCIe bus, we overcome bandwidth bottlenecks and improve LLM serving throughput on high-performance GPUs like the RTX 5080.

## 📁 Project Structure
```text
PolyU_COMP_Final_Year_Project_2026_Spring/
├── benchmarks/               # Python performance profiling scripts
├── docker/                   # Docker environment (Dockerfile, docker-compose, run.sh)
├── final_report/             # Final FYP thesis (PDF)
├── integration/              # Core source code
│   ├── compression_pipeline/ # Python Swap Manager logic
│   └── cuszp_wrapper/        # C++/PyBind11 source & CMakeLists.txt
├── interim_report/           # Mid-term progress documentation
├── project_proposal/         # Initial design and scope
├── .gitattributes            # Forces LF line endings for .sh files
├── run.sh                    # 🚀 Main Automation Script (Compiles & Benchmarks)
└── README.md                 # This comprehensive guide
```

---

## 🛠️ Step 1: Pre-requisites & System Preparation

### 1. Hardware Requirement
* **NVIDIA GPU** with CUDA support is required for the compression kernels.

### 2. Platform-Specific Setup
| Platform | Requirement |
| :--- | :--- |
| **Linux (Ubuntu)** | Install NVIDIA Drivers and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). |
| **Windows** | You **MUST** use **WSL2** (Ubuntu). Install Docker Desktop and enable WSL2 integration. Native PowerShell/CMD will fail to build C++ components. |
| **macOS** | You can view code and reports, but **GPU execution is impossible**. Building the C++ wrapper will only work in a "mock" CPU mode. |

---

## 🐍 Step 2: Python Virtual Environment (venv)

Always use a virtual environment to avoid corrupting system libraries.

```bash
# 1. Create the environment
python3 -m venv venv

# 2. Activate the environment
# For Linux/WSL2/macOS:
source venv/bin/activate
```

---

## 🚀 Step 3: Detailed Execution Flow

You have two ways to run this project: **Docker (Recommended)** or **Native**.

### Option A: Running via Docker (Recommended)
Docker ensures the CUDA, PyTorch, and PyBind11 versions are perfectly matched.

1.  **Navigate to the docker folder:**
    ```bash
    cd docker
    ```
2.  **Start and Enter the Container:**
    ```bash
    sudo ./run.sh run
    ```
3.  **Inside the Container (Workspace):**
    The container automatically maps the root to `/workspace`. Run the main script:
    ```bash
    # You are now at root@container:/workspace#
    sudo ./run.sh
    ```

### Option B: Running Natively
Only use this if you have CUDA 12.1+ and PyTorch 2.1+ installed locally.

1.  **Run the root script directly:**
    ```bash
    sudo ./run.sh
    ```

---

## 🔍 Step 4: What the Automation Script Does
When you run `sudo ./run.sh`, the following sequence occurs automatically:
1.  **C++ Compilation:** Navigates to `integration/cuszp_wrapper`, creates a `build` directory, and runs `cmake` and `make` to generate `cuszp_wrapper_cpp.so`.
2.  **Environment Check:** Verifies if the `.so` file is correctly placed in `integration/compression_pipeline`.
3.  **Baseline Profiling:** Runs `benchmarks/baseline_profiling.py` to measure raw PCIe H2D/D2H speeds.
4.  **Compression Benchmark:** Runs `benchmarks/compression_benchmark.py` to calculate the throughput (GB/s) and compression ratio of the cuSZp kernels.

---

## ❓ Troubleshooting & Common Errors

### 1. Permission Denied
If you see `Permission denied`, you missed the `chmod` step. Run:
`sudo chmod +x run.sh docker/run.sh`

### 2. Windows Line Ending Error (`^M` or `bad interpreter`)
Windows Git often changes `LF` to `CRLF`.
* **Fix:** In VS Code, look at the bottom right status bar. If it says **CRLF**, click it and change it to **LF**. Save the file and try again.

### 3. `ModuleNotFoundError: No module named 'cuszp_wrapper_cpp'`
This happens if the C++ build failed or the `PYTHONPATH` is not set.
* **Fix:** Our `run.sh` automatically handles `PYTHONPATH`. If running manually, use:
    `export PYTHONPATH=$PYTHONPATH:$(pwd)/integration/compression_pipeline`

### 4. GPU Not Found in Docker
Run `nvidia-smi` on your host. If it works there but fails in Docker, you need to reinstall the **NVIDIA Container Toolkit**.

---

## 📑 Project Documentation & Reports
* **Final Report:** [final_report/FYP_2026_Final_Report.pdf](final_report/FYP_2026_Final_Report.pdf)
* **Interim Report:** [interim_report/FYP_2026_Interim_Report.pdf](interim_report/FYP_2026_Interim_Report.pdf)
* **Initial Proposal:** [project_proposal/Compress_Transfer_Decompress.pdf](project_proposal/Compress_Transfer_Decompress.pdf)