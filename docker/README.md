# Docker Environment Setup & Usage Guide

To ensure a consistent compilation environment for the cuSZp C++ extension and avoid CUDA/PyTorch version conflicts (Dependency Hell) on the host machine, this project strictly requires development and testing to be conducted within the provided Docker container.

## 🛠️ Prerequisites & Configuration

### Linux Users (Recommended & Easiest)

1. **Install Docker**:
   ```bash
   curl -fsSL [https://get.docker.com](https://get.docker.com) -o get-docker.sh && sh get-docker.sh
   ```
2. **Install NVIDIA Container Toolkit** (Crucial step to allow containers to access the GPU):
   Follow the [Official NVIDIA Documentation](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) to configure the apt repository, run `sudo apt-get install -y nvidia-container-toolkit`, and then restart the Docker service.

---

### Windows (WSL2) User Guide

Developing C++ CUDA extensions using Docker on Windows is prone to pitfalls. Please strictly follow these steps:

1. **Configure the Base WSL2 Environment**
   Run PowerShell as Administrator and execute `wsl --install -d Ubuntu`.
2. **Install Windows NVIDIA Drivers**
   Install the latest GPU drivers directly on your Windows host machine. **Do not attempt to install Linux drivers inside WSL2 Ubuntu**. If you can see the GPU details by running `nvidia-smi` in the WSL2 terminal, the driver is working correctly.
3. **Configure Docker Desktop**
   Install Docker Desktop, go to `Settings > Resources > WSL Integration`, and ensure integration is enabled for your installed Ubuntu distribution.
4. **Verify GPU Passthrough**
   Execute the following in your WSL2 terminal:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
   ```
   If it outputs your GPU status normally, the configuration is perfect.

---

## 💻 Container Lifecycle Management

We provide a convenient infrastructure script, `docker/run.sh`, to control the container lifecycle. Execute it **from within the `docker/` directory**:

| Command | Description |
|---------|-------------|
| `./run.sh build` | **(Rarely Used)** Builds the base image. Since it sets up CUDA 12.1.1 and PyTorch dependencies, the initial build may be slow. You usually only need to run this if you modify system-level dependencies in the `Dockerfile`. |
| `./run.sh run` | **(Daily Use)** Starts and enters an interactive container. This command mounts the project root directory from your host machine to `/workspace` inside the container. Any code you write or `.so` files you compile inside the container will be synchronized directly to your local Windows/Linux disk. |

> 💡 **Best Practice**: We do **not** recommend using `docker/run.sh build` to pack your project code directly into the image. The correct development workflow for this project is: Use `docker/run.sh run` to enter the container with the base environment, then directly call the root-level script `./run.sh` from the `/workspace` directory for "just-in-time compilation and execution."

## ❓ Troubleshooting (FAQ)

### Q1: What should I do if I get `cannot find -ltorch_python` during C++ extension compilation?
**A**: This usually happens because CMake cannot find the internal library path of PyTorch. Please ensure you are using the root-level `./run.sh` to compile, as this script has fixed absolute path mounting issues via environment variables and CMake flags.

### Q2: Why does it say `ModuleNotFoundError` when I `import cuszp_wrapper_cpp`?
**A**: This is the classic "Docker mount masking" issue. If you haven't compiled the `.so` file on your host machine, and the `/workspace` inside the container is overwritten by the empty directory from your host, Python won't find the module.
**Solution**: Upon entering the container, immediately run `./run.sh` (the one located in the root directory `/workspace`) to generate the compilation artifacts directly in the container's mounted directory.

### Q3: Why do I get `bad interpreter` or command not found when running the root `./run.sh` on Windows?
**A**: This is because Windows automatically changes the script's line endings to `CRLF`.
**Solution**: This project has locked line endings to `LF` via `.gitattributes`. If you still encounter this issue, please manually switch the script's line endings back to `LF` in the bottom right corner of VS Code.