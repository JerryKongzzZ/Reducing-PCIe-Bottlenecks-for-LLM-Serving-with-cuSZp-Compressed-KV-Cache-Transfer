# Dockerfile for vLLM with cuSZp integration
# This Dockerfile is used to build and run the project in a CUDA-supported environment

FROM docker.io/nvidia/cuda:12.1.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# Install system dependencies
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

# Install Python dependencies
RUN pip3 install --upgrade pip
RUN pip3 install \
    torch \
    numpy \
    pybind11 \
    setuptools \
    wheel \
    vllm

# Set working directory
WORKDIR /workspace

# Copy project files (excluding cuSZp directory as it will be cloned from GitHub)
COPY . /workspace/

# Clone cuSZp repository (delete if directory already exists)
RUN rm -rf /opt/cuSZp && \
    git clone https://github.com/szcompressor/cuSZp.git /opt/cuSZp

# Compile cuSZp
WORKDIR /opt/cuSZp
RUN mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=../install/ .. && \
    make -j$(nproc) && \
    make install

# Compile cuSZp wrapper
WORKDIR /workspace/integration/cuszp_wrapper
RUN mkdir -p build && cd build && \
    cmake .. \
      -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
      -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
      -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')" && \
    make -j$(nproc)

# Return to working directory
WORKDIR /workspace

# Set default command
CMD ["/bin/bash"]

