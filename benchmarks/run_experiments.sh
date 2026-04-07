#!/bin/bash
set -e

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$WORKSPACE_ROOT"

# Ensure environment is clean and built
cd "$WORKSPACE_ROOT/integration/cuszp_wrapper"
rm -rf build && mkdir -p build && cd build
cmake .. -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"
make -j$(nproc)
cd "$WORKSPACE_ROOT"

MODELS=(
    "gpt2"
    "Qwen/Qwen2.5-0.5B"
    "Qwen/Qwen2.5-1.5B"
    "facebook/opt-125m"
    "facebook/opt-350m"
    "EleutherAI/pythia-160m"
    "EleutherAI/pythia-410m"
    "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
)

mkdir -p data/

for MODEL in "${MODELS[@]}"; do
    SAFE_NAME=$(echo "$MODEL" | tr '/' '_')
    echo "=========================================="
    echo "🧪 Running Experiment for: $MODEL"
    echo "=========================================="
    
    python3 benchmarks/generate_real_kv_cache.py --model "$MODEL" --output "data/${SAFE_NAME}_kv_cache.pt"
    
    PYTHONPATH=integration/compression_pipeline python3 benchmarks/compression_benchmark.py \
        --tensor-size 4194304 \
        --scan-eb \
        --use-real-kv "data/${SAFE_NAME}_kv_cache.pt" \
        --output "data/${SAFE_NAME}_results.json"
done

echo "✅ All experiments completed! Results saved to data/"
