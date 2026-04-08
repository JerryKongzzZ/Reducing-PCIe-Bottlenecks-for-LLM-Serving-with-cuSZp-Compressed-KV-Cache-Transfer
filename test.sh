#!/bin/bash
# Stop execution immediately if any command fails
set -e 

# Get the directory of the current script as the workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "🚀 [1/3] Automatically compiling cuSZp C++ core extension..."
echo "=========================================="
# Enter the build directory
cd "$WORKSPACE_ROOT/integration/cuszp_wrapper"

# Clean old cache and recompile
rm -rf build_local && mkdir -p build_local && cd build_local
cmake .. \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
  -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"
make -j$(nproc)

echo "✅ C++ extension compilation completed!"
echo ""

echo "=========================================="
echo "📊 [2/3] Running Unified Benchmark Pipeline for all models..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
# Automatically inject module path and run unified test
PYTHONPATH=integration/compression_pipeline python3 benchmarks/benchmark_pipeline.py --tensor-size 4194304 --iterations 50
echo ""

echo "=========================================="
echo "🔌 [3/3] Testing vLLM CacheEngine Monkey Patch Integration..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
python3 benchmarks/test_vllm_integration.py
echo ""

echo "=========================================="
echo "🎉 Congratulations! All Benchmark tests have been executed successfully!"
echo "📂 The test results have been successfully saved to .json files."
