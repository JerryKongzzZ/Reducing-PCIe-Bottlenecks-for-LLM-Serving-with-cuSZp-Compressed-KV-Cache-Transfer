#!/bin/bash
# 遇到任何报错立即停止执行
set -e 

echo "=========================================="
echo "🚀 [1/3] 正在自动化编译 cuSZp C++ 核心扩展..."
echo "=========================================="
# 确保在容器的根目录工作
cd /workspace/integration/cuszp_wrapper

# 清理旧缓存并重新编译
rm -rf build && mkdir -p build && cd build
cmake .. \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
  -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"
make -j$(nproc)

echo "✅ C++ 扩展编译完成！"
echo ""

echo "=========================================="
echo "📊 [2/3] 开始执行基线性能分析 (Baseline)..."
echo "=========================================="
cd /workspace
# 自动注入模块路径并运行基线测试
PYTHONPATH=integration/compression_pipeline python3 benchmarks/baseline_profiling.py --device-id 0 --iterations 50
echo ""

echo "=========================================="
echo "🗜️ [3/3] 开始执行 cuSZp 压缩性能测试..."
echo "=========================================="
# 自动注入模块路径并运行压缩测试
PYTHONPATH=integration/compression_pipeline python3 benchmarks/compression_benchmark.py --tensor-size 1048576 --error-bound 1e-4 --iterations 50
echo ""

echo "=========================================="
echo "🎉 恭喜！所有 Benchmark 性能测试执行完毕！"
echo "📂 测试结果已成功保存至 .json 文件中。"
echo "=========================================="