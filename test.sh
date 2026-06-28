#!/bin/bash
# Stop execution immediately if any command fails
set -e 

# Use HF mirror for faster model downloading
export HF_ENDPOINT=https://hf-mirror.com

# Get the directory of the current script as the workspace root
WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================="
echo "🚀 [1/7] Automatically compiling cuSZp C++ core extension..."
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
echo "📊 [2/7] Running Unified Benchmark Pipeline for all models..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
# Automatically inject module path and run unified test
PYTHONPATH=integration/compression_pipeline python3 benchmarks/benchmark_pipeline.py --tensor-size 4194304 --iterations 50
echo ""

echo "=========================================="
echo "🎯 [3/7] Generating Offline Layer Sensitivity Profile..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
PYTHONPATH=integration/compression_pipeline python3 benchmarks/layer_sensitivity_sweep.py --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2
echo ""

echo "=========================================="
echo "📈 [4/7] End-to-End Policy Evaluation (Adaptive vs Static)..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
PYTHONPATH=integration/compression_pipeline python3 benchmarks/evaluate_policies.py --models gpt2 --out data/eval_summary.json
echo ""

echo "=========================================="
echo "🖼️ [5/7] Generating Paper Figures (Pareto, Queue, Ablation)..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
PYTHONPATH=integration/compression_pipeline python3 benchmarks/run_pareto_queue_ablation.py
echo ""

echo "=========================================="
echo "🔌 [6/7] Testing vLLM CacheEngine Monkey Patch Integration..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
python3 benchmarks/test_vllm_integration.py
echo ""

echo "=========================================="
echo "📊 [7/7] Generating Summary Plots..."
echo "=========================================="
cd "$WORKSPACE_ROOT"
python3 benchmarks/plot_summary.py
echo ""

echo "=========================================="
echo "🎉 Congratulations! All tests and artifact generation have been executed successfully!"
echo "📂 The test results and figures have been successfully saved to the data/ folder."
