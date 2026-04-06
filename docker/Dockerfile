# Dockerfile for vLLM with cuSZp integration
# 这个Dockerfile用于在支持CUDA的环境中构建和运行项目

FROM nvidia/cuda:11.8.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    cmake \
    build-essential \
    python3.10 \
    python3.10-dev \
    python3-pip \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 安装Python依赖
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch \
    numpy \
    pybind11 \
    setuptools \
    wheel \
    vllm

# 设置工作目录
WORKDIR /workspace

# 复制项目文件（排除cuSZp目录，因为将从GitHub克隆）
COPY . /workspace/

# 克隆cuSZp仓库（如果目录已存在则先删除）
RUN rm -rf /workspace/cuSZp && \
    git clone https://github.com/szcompressor/cuSZp.git /workspace/cuSZp

# 编译cuSZp
WORKDIR /workspace/cuSZp
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ .. && \
    make -j$(nproc) && \
    make install

# 编译cuSZp包装器
WORKDIR /workspace/integration/cuszp_wrapper
RUN mkdir -p build && cd build && \
    cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && \
    make -j$(nproc)

# 返回工作目录
WORKDIR /workspace

# 设置默认命令
CMD ["/bin/bash"]

