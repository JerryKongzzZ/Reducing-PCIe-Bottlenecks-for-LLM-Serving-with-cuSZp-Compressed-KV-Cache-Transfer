cmake_minimum_required(VERSION 3.21)
project(cuszp_wrapper)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 查找依赖
find_package(CUDA REQUIRED)
find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
find_package(pybind11 REQUIRED)
# 👉 新增：查找 Torch 库
find_package(Torch REQUIRED)

# 查找cuSZp
# 假设cuSZp安装在 ../cuSZp/install
set(CUSZP_ROOT "${CMAKE_SOURCE_DIR}/../../cuSZp")
set(CUSZP_INSTALL_DIR "${CUSZP_ROOT}/install")

# cuSZp头文件路径
set(CUSZP_INCLUDE_DIRS
    "${CUSZP_ROOT}/include"
    "${CUSZP_INSTALL_DIR}/include"
)

# cuSZp库路径
set(CUSZP_LIB_DIRS
    "${CUSZP_INSTALL_DIR}/lib"
)

# 查找cuSZp库
find_library(CUSZP_LIB
    NAMES cuSZp libcuSZp
    PATHS ${CUSZP_LIB_DIRS}
    REQUIRED
)

# 包含目录
include_directories(
    ${CUSZP_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${pybind11_INCLUDE_DIRS}
)

# 源文件
set(SOURCES
    cuszp_wrapper.cpp
    pybind11_bindings.cpp
)

# 创建Python模块
pybind11_add_module(cuszp_wrapper_cpp ${SOURCES})

# 链接库
target_link_libraries(cuszp_wrapper_cpp PRIVATE
    ${CUSZP_LIB}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
    ${TORCH_LIBRARIES}
    torch_python
)

# 编译选项
target_compile_options(cuszp_wrapper_cpp PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra>
    ${TORCH_CXX_FLAGS}    # 👉 新增：应用 Torch 推荐的编译 Flag
)

# 设置输出目录
set_target_properties(cuszp_wrapper_cpp PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/../compression_pipeline"
    PREFIX ""
)

# 安装规则（可选）
install(TARGETS cuszp_wrapper_cpp
    LIBRARY DESTINATION lib
)

