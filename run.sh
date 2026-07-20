#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/venv/bin/python}"
SKIP_BUILD="${SKIP_BUILD:-1}"
RUN_VLLM_SMOKE="${RUN_VLLM_SMOKE:-0}"
# CUDA 12.0 cannot name RTX 5080's native architecture. Embed compute_90 PTX
# so the Blackwell driver can JIT the custom batched decoder. Override this
# value when building for a different deployment target.
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"
export TORCH_CUDA_ARCH_LIST

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python environment not found: ${PYTHON_BIN}" >&2
  exit 1
fi

export PYTHONPATH="${WORKSPACE_ROOT}/integration/compression_pipeline:${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

if [[ "${SKIP_BUILD}" != "1" ]]; then
  BUILD_DIR="${WORKSPACE_ROOT}/integration/cuszp_wrapper/build_local"
  cmake -S "${WORKSPACE_ROOT}/integration/cuszp_wrapper" -B "${BUILD_DIR}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    -DPython3_NumPy_INCLUDE_DIRS="$(${PYTHON_BIN} -c 'import numpy; print(numpy.get_include())')" \
    -Dpybind11_DIR="$(${PYTHON_BIN} -m pybind11 --cmakedir)" \
    -DCMAKE_PREFIX_PATH="$(${PYTHON_BIN} -c 'import torch; print(torch.utils.cmake_prefix_path)')"
  cmake --build "${BUILD_DIR}" -j"$(nproc)"
fi

cd "${WORKSPACE_ROOT}"
"${PYTHON_BIN}" -m pytest -q

if [[ "${RUN_VLLM_SMOKE}" == "1" ]]; then
  MODEL="${MODEL:-Qwen/Qwen2.5-0.5B}"
  METRICS="${METRICS:-data/vllm_offload_smoke.jsonl}"
  SUMMARY="${SUMMARY:-data/vllm_offload_smoke_summary.json}"
  CUSZP_MODE="${CUSZP_MODE:-fixed}"
  ERROR_BOUND="${ERROR_BOUND:-1e-5}"
  "${PYTHON_BIN}" benchmarks/smoke_vllm_compressed_offload.py \
    --model "${MODEL}" \
    --metrics "${METRICS}" \
    --summary "${SUMMARY}" \
    --codec cuszp \
    --cuszp-mode "${CUSZP_MODE}" \
    --error-bound "${ERROR_BOUND}" \
    --async-store \
    --profile-restore-stages \
    --batch-restore-transfers
fi

echo "Maintained validation workflow completed."
