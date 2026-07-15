#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export PYTHONPATH="${WORKSPACE_ROOT}/integration/compression_pipeline:${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

BENCHMARK_ITERATIONS="${BENCHMARK_ITERATIONS:-5}"
EVAL_ITERATIONS="${EVAL_ITERATIONS:-5}"
RUN_SYNTHETIC="${RUN_SYNTHETIC:-0}"
SKIP_BUILD="${SKIP_BUILD:-0}"

if [[ -x "${WORKSPACE_ROOT}/venv/bin/python3.12" ]]; then
  PYTHON_BIN="${WORKSPACE_ROOT}/venv/bin/python3.12"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "Python 3 was not found. Please create a virtual environment or install Python 3 first." >&2
  exit 1
fi

mkdir -p "${WORKSPACE_ROOT}/data"

echo "=========================================="
echo "🚀 [1/5] Preparing the compression runtime..."
echo "=========================================="
if [[ "${SKIP_BUILD}" != "1" ]]; then
  cd "${WORKSPACE_ROOT}/integration/cuszp_wrapper"
  rm -rf build_local && mkdir -p build_local
  cd build_local
  if ! cmake .. \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    -DPython3_NumPy_INCLUDE_DIRS="$(${PYTHON_BIN} -c 'import numpy; print(numpy.get_include())')" \
    -Dpybind11_DIR="$(${PYTHON_BIN} -m pybind11 --cmakedir)" \
    -DCMAKE_PREFIX_PATH="$(${PYTHON_BIN} -c "import torch; print(torch.utils.cmake_prefix_path)")" \
    -DCMAKE_CXX_FLAGS="$(${PYTHON_BIN} -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"; then
    echo "⚠️  cuSZp build configuration failed; continuing in CPU/zlib fallback mode."
  elif ! cmake --build . -j"$(nproc)"; then
    echo "⚠️  cuSZp build failed; continuing in CPU/zlib fallback mode."
  else
    echo "✅ Extension build completed"
  fi
else
  echo "⚠️  Skipping extension build because SKIP_BUILD=1"
fi

echo ""
echo "=========================================="
echo "📊 [2/5] Running the unified benchmark pipeline..."
echo "=========================================="
cd "${WORKSPACE_ROOT}"
if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  "${PYTHON_BIN}" benchmarks/benchmark_pipeline.py --tensor-size 4194304 --iterations "${BENCHMARK_ITERATIONS}" --synthetic
else
  "${PYTHON_BIN}" benchmarks/benchmark_pipeline.py --tensor-size 4194304 --iterations "${BENCHMARK_ITERATIONS}"
fi

echo ""
echo "=========================================="
echo "🎯 [3/5] Profiling layer sensitivity..."
echo "=========================================="
cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" benchmarks/layer_sensitivity_sweep.py --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2

echo ""
echo "=========================================="
echo "📈 [4/5] Evaluating compression policies..."
echo "=========================================="
cd "${WORKSPACE_ROOT}"
if [[ "${RUN_SYNTHETIC}" == "1" ]]; then
  "${PYTHON_BIN}" benchmarks/evaluate_policies.py --out data/eval_summary.json --iterations "${EVAL_ITERATIONS}" --synthetic
else
  "${PYTHON_BIN}" benchmarks/evaluate_policies.py --out data/eval_summary.json --iterations "${EVAL_ITERATIONS}"
fi

echo ""
echo "=========================================="
echo "🖼️ [5/5] Generating summary figures..."
echo "=========================================="
cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" benchmarks/run_pareto_queue_ablation.py
"${PYTHON_BIN}" benchmarks/plot_summary.py

echo ""
echo "🎉 Workflow completed. Outputs are available under data/"
