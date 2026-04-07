#!/bin/bash
# Stop execution immediately if any command fails
set -e 

# Get the directory of the current script as the workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "🚀 [1/5] Automatically compiling cuSZp C++ core extension..."
echo "=========================================="
# Enter the build directory
cd "$WORKSPACE_ROOT/integration/cuszp_wrapper"

# Clean old cache and recompile
rm -rf build && mkdir -p build && cd build
cmake .. \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
  -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"
make -j$(nproc)

echo "✅ C++ extension compilation completed!"
echo ""

echo "=========================================="
echo "🧠 [2/5] Generating Real KV Cache Data (Llama-3 or GPT2)..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
# Generate real KV cache
python3 benchmarks/generate_real_kv_cache.py --model "gpt2" --output "data/real_kv_cache.pt"
echo "✅ Real KV Cache generation completed!"
echo ""

echo "=========================================="
echo "📊 [3/5] Starting baseline performance analysis..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
# Automatically inject module path and run baseline test
PYTHONPATH=integration/compression_pipeline python3 benchmarks/baseline_profiling.py --device-id 0 --iterations 50
echo ""

echo "=========================================="
echo "🗜️ [4/5] Starting cuSZp compression performance test (Real Data)..."
echo "=========================================="
# Automatically inject module path and run compression test with real KV cache
PYTHONPATH=integration/compression_pipeline python3 benchmarks/compression_benchmark.py --tensor-size 1048576 --error-bound 1e-4 --iterations 50 --use-real-kv data/real_kv_cache.pt
echo ""

echo "=========================================="
echo "🔌 [5/5] Testing vLLM CacheEngine Monkey Patch Integration..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
python3 benchmarks/test_vllm_integration.py
echo ""

echo "=========================================="
echo "🎉 Congratulations! All Benchmark tests have been executed successfully!"
echo "📂 The test results have been successfully saved to .json files."
