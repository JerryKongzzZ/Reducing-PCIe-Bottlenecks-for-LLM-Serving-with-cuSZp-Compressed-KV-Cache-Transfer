# Docker Deployment Guide (vLLM + cuSZp)

This guide provides instructions for running the project within a Docker container. This is the **recommended** method to ensure a consistent environment with the correct CUDA and PyTorch versions.

## 🛠️ Prerequisites

### 1. NVIDIA Container Toolkit (Mandatory)
To allow Docker to access your GPU, you must install the NVIDIA Container Toolkit.
* **Linux:** Follow the [official NVIDIA guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
* **Windows:** Install **Docker Desktop** and enable the **WSL2 backend**. No extra toolkit is needed inside WSL2 as Docker Desktop handles the passthrough.

### 2. Permissions
Docker commands and scripts often require root privileges. Always prefix commands with `sudo` unless your user is in the `docker` group.

```bash
# Grant execution permission to the docker helper script
sudo chmod +x run.sh
```

## 🚀 Container Lifecycle

Navigate to this directory (`docker/`) and use the helper script:

| Task | Command |
| :--- | :--- |
| **Build Image** | `sudo ./run.sh build` |
| **Start & Enter** | `sudo ./run.sh run` |
| **Cleanup** | `sudo ./run.sh stop` |

### 🏁 Step-by-Step Execution

**1. Enter the Container:**
```bash
sudo ./run.sh run
```
Your project root is now mapped to `/workspace` inside the container.

**2. Setup Virtual Environment (Inside Container):**
The container comes with system-level PyTorch, but we still use a `venv` for isolation:
```bash
# Inside the container terminal:
cd /workspace
python3 -m venv venv
source venv/bin/activate
```

**3. Run the Automated Build & Benchmark:**
Inside the container, run the root `run.sh` to compile the C++ wrappers and start the benchmarks:
```bash
# Inside the container, with venv activated:
sudo ./run.sh
```

## ⚠️ Cross-Platform Troubleshooting

### Windows (WSL2) Line Endings
If you receive `bash: ./run.sh: /bin/bash^M: bad interpreter` on Windows, Git has likely converted line endings to `CRLF`.
* **Fix:** Open the file in VS Code and change the EOL (bottom right) from `CRLF` to `LF`, then save.

### Permission Denied inside `/workspace`
Since the files are mounted from your host machine, they might be owned by your host user. 
* **Fix:** Use `sudo chown -R root:root /workspace` inside the container if the build system cannot write to the `build/` folders.

### macOS (NVIDIA Limitation)
Docker on macOS **cannot** access the GPU because Apple hardware does not support NVIDIA CUDA. If you are on a Mac, you must use a remote Linux server or a Windows PC with an NVIDIA GPU.