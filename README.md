# Compress-Transfer-Decompress for LLM Serving: cuSZp-Enabled CPU-GPU Data Pipeline in vLLM

## 🌟 Project Overview
This project is part of the Final Year Project (FYP) 2026 at **The Hong Kong Polytechnic University**. 
The goal is to integrate the **cuSZp** error-bounded lossy compression framework into **vLLM** to optimize KV Cache swapping between CPU and GPU. By compressing data before transferring it over the PCIe bus, we aim to overcome bandwidth bottlenecks and improve LLM serving throughput.

## 📁 Project Structure
```text
POLYU_COMP_FIN.../
├── benchmarks/               # Performance profiling scripts
├── docker/                   # Docker deployment (See docker/README.md)
├── final_report/             # Final FYP thesis and documentation
├── integration/              # Core source code (C++ Wrappers & Python Pipeline)
├── interim_report/           # Mid-term progress reports
├── project_proposal/         # Initial project scope and design
├── .gitattributes            # Git configuration for LF line endings
├── run.sh                    # 🚀 Root automation script (Build & Test)
└── README.md                 # Project entry point
```

## 💻 Native Setup (Non-Docker)

### Prerequisites & Permissions
Before starting, ensure you have the correct permissions. On Linux/macOS, use `sudo` to grant execution rights to the automation scripts to avoid `Permission Denied` errors.

**1. Grant Permissions:**
```bash
sudo chmod +x run.sh
sudo chmod +x docker/run.sh
```

**2. Platform-Specific Requirements:**
* **Linux (Recommended):** Ensure NVIDIA Drivers and CUDA Toolkit (12.1+) are installed.
* **Windows:** Use **WSL2** (Ubuntu). Native Windows CMD/PowerShell is not supported for the C++ build chain.
* **macOS:** Building the wrappers is possible, but **GPU execution (CUDA) is not supported** on macOS. Testing can only be done in "Mock" mode.

### 🐍 Python Virtual Environment (venv)
To keep the system clean and avoid dependency conflicts, always use a virtual environment:

```bash
# Create the environment
python3 -m venv venv

# Activate it
# On Linux/macOS/WSL2:
source venv/bin/activate
# On Windows (if using native python, though WSL2 is preferred):
.\venv\Scripts\activate

# Install core dependencies
sudo ./venv/bin/pip install -r requirements.txt
```

### 🚀 Running the Pipeline
Once the environment is active, run the root script to compile and benchmark:
```bash
sudo ./run.sh
```

## 📄 Project Reports
* **Proposal:** [Project_Proposal.pdf](project_proposal/Compress_Transfer_Decompress.pdf)
* **Interim:** [FYP_2026_Interim_Report.pdf](interim_report/FYP_2026_Interim_Report.pdf)
* **Final:** [FYP_2026_Final_Report.pdf](final_report/FYP_2026_Final_Report.pdf)

---
*For Docker-based deployment, please refer to the dedicated guide in [docker/README.md](docker/README.md).*