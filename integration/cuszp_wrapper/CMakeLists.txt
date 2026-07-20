cmake_minimum_required(VERSION 3.21)
project(cuszp_wrapper LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find dependencies
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
# 👉 New: Find Torch library
find_package(Torch REQUIRED)
# 👉 New: Explicitly find torch_python library in PyTorch's install directory
find_library(TORCH_PYTHON_LIB torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

# Find cuSZp
# cuSZp installed in /opt/cuSZp/install
set(CUSZP_ROOT "/opt/cuSZp")
set(CUSZP_INSTALL_DIR "${CUSZP_ROOT}/install")

# cuSZp include paths
set(CUSZP_INCLUDE_DIRS
    "${CUSZP_ROOT}/include"
    "${CUSZP_INSTALL_DIR}/include"
)

# cuSZp library paths
set(CUSZP_LIB_DIRS
    "${CUSZP_INSTALL_DIR}/lib"
)

# Find cuSZp library
find_library(CUSZP_LIB
    NAMES cuSZp libcuSZp
    PATHS ${CUSZP_LIB_DIRS}
    REQUIRED
)

# Include directories
include_directories(
    ${CUSZP_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# Source files
set(SOURCES
    cuszp_wrapper.cpp
    cuszp_fastpath.cu
    cuszp_batched_fixed.cu
    pybind11_bindings.cpp
)

# Create Python module
pybind11_add_module(cuszp_wrapper_cpp ${SOURCES})

# Link libraries
target_link_libraries(cuszp_wrapper_cpp PRIVATE
    ${CUSZP_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIB}    # 👉 Modified: Use absolute path variable
)

# Compile options
target_compile_options(cuszp_wrapper_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    ${TORCH_CXX_FLAGS}    # 👉 New: Apply Torch's recommended compilation flags
)

# Set output directory
set_target_properties(cuszp_wrapper_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../compression_pipeline"
    PREFIX ""
    CUDA_STANDARD 17
    CUDA_STANDARD_REQUIRED ON
)

# Installation rules (optional)
install(TARGETS cuszp_wrapper_cpp
    LIBRARY DESTINATION lib
)
