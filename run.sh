#!/usr/bin/env bash
set -euo pipefail

WORKSPACE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${WORKSPACE_ROOT}/venv/bin/python}"
CUSZP_ROOT="${CUSZP_ROOT:-/opt/cuSZp}"
SKIP_BUILD="${SKIP_BUILD:-1}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-9.0+PTX}"

export CUSZP_ROOT TORCH_CUDA_ARCH_LIST
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export PYTHONPATH="${WORKSPACE_ROOT}/integration/compression_pipeline:${WORKSPACE_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

usage() {
  cat <<'EOF'
Usage: ./run.sh [validate|dry-run|calibrate|calibrate-fresh|experiments|rerun|assets|verify-paper|full]

  validate         Verify pinned software/results, optionally build, and test.
  dry-run          Validate every formal top-level command without inference.
  calibrate        Rebuild the merged/mixed profile from admitted prompt profiles.
  calibrate-fresh  Rerun all eight GPU layer-sensitivity sweeps, then rebuild.
  experiments      Resume canonical GPU experiments; this can take many hours.
  rerun            Run every paper experiment in a new independent output root.
  assets           Regenerate tables/figures and sync marked paper result blocks.
  verify-paper     Verify hardware/models/hashes and paper/data consistency.
  full             Validate, calibrate, resume experiments, and regenerate assets.

Environment:
  PYTHON_BIN=/path/to/python
  CUSZP_ROOT=/path/to/pinned/cuSZp
  SKIP_BUILD=0
  HF_HUB_OFFLINE=0
  REPRO_OUT_ROOT=data/reproductions/run-01   required by rerun
EOF
}

require_python() {
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python environment not found: ${PYTHON_BIN}" >&2
    exit 1
  fi
}

build_native() {
  if [[ "${SKIP_BUILD}" == "1" ]]; then
    return
  fi
  local build_dir="${WORKSPACE_ROOT}/integration/cuszp_wrapper/build_local"
  cmake -S "${WORKSPACE_ROOT}/integration/cuszp_wrapper" -B "${build_dir}" \
    -DPython3_EXECUTABLE="${PYTHON_BIN}" \
    -DPython3_NumPy_INCLUDE_DIRS="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.get_include())')" \
    -Dpybind11_DIR="$("${PYTHON_BIN}" -m pybind11 --cmakedir)" \
    -DCMAKE_PREFIX_PATH="$("${PYTHON_BIN}" -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
    -DCUSZP_ROOT="${CUSZP_ROOT}"
  cmake --build "${build_dir}" -j"$(nproc)"
}

verify_software() {
  "${PYTHON_BIN}" reproducibility/verify_environment.py \
    --scope software --check-results
}

verify_full() {
  "${PYTHON_BIN}" reproducibility/verify_environment.py \
    --scope full --check-results
}

validate() {
  verify_software
  build_native
  "${PYTHON_BIN}" -m pytest -q
}

calibrate_adaptive() {
  local fresh="${1:-0}"
  local sweep_args=()
  if [[ "${fresh}" == "0" ]]; then
    sweep_args+=(--aggregate-only)
  fi
  "${PYTHON_BIN}" benchmarks/build_multi_prompt_sensitivity_profile.py \
    --model Qwen/Qwen2.5-1.5B \
    --prompts-file data/infocom_calibration_prompts.json \
    --profile-dir data/qwen2.5-1.5b_multi_prompt_sensitivity_1e-5_1e-4_per_prompt \
    --out data/qwen2.5-1.5b_multi_prompt_sensitivity_top1-0.875.json \
    --eps 1e-5 1e-4 --probe-tokens 8 \
    --kl-threshold 0.01 --min-top1-match 0.875 \
    "${sweep_args[@]}"
  "${PYTHON_BIN}" benchmarks/select_mixed_bound_profile.py \
    --profile data/qwen2.5-1.5b_multi_prompt_sensitivity_top1-0.875.json \
    --out data/qwen2.5-1.5b_mixed_21x1e-4_7x1e-5.json \
    --tight-bound 1e-5 --loose-bound 1e-4 --loose-layers 21
}

run_quality_suite() {
  local out_dir="$1"
  local resume="$2"
  local dry="${3:-0}"
  local extra=()
  [[ "${resume}" == "1" ]] && extra+=(--resume)
  [[ "${dry}" == "1" ]] && extra+=(--dry-run)
  set +e
  "${PYTHON_BIN}" benchmarks/run_vllm_repeated_smoke.py \
    --methods async_raw async_cuszp_1e-5 async_zstd async_lz4 async_int8 \
    --trials 5 --trial-order interleaved --out-dir "${out_dir}" \
    --model Qwen/Qwen2.5-1.5B \
    --max-model-len 4096 --kv-cache-memory-bytes 134217728 \
    --prompt-repeats 182 182 182 182 182 182 182 182 \
    --prompt-style shared \
    --prompt-file data/infocom_long_context_quality_v1/prompts.json \
    --batch-prompts --batch-restore-transfers \
    --error-bound 1e-4 --cuszp-mode fixed \
    --cpu-offload-gb 4 --gpu-memory-utilization 0.8 --max-tokens 16 \
    --quality-gate --quality-baseline-method async_raw \
    --quality-min-task-accuracy 0.875 --quality-max-task-accuracy-drop 0 \
    "${extra[@]}"
  local status="$?"
  set -e
  if [[ "${status}" != "0" && "${status}" != "2" ]]; then
    return "${status}"
  fi
  if [[ "${status}" == "2" ]]; then
    echo "Quality gate rejected at least one comparison codec as expected; evidence was retained."
  fi
}

run_adaptive_pilot() {
  local out_dir="$1"
  local resume="$2"
  local dry="${3:-0}"
  local extra=()
  [[ "${resume}" == "1" ]] && extra+=(--resume)
  [[ "${dry}" == "1" ]] && extra+=(--dry-run)
  "${PYTHON_BIN}" benchmarks/run_vllm_repeated_smoke.py \
    --methods async_raw async_cuszp_1e-5 async_adaptive \
    --trials 2 --trial-order interleaved --out-dir "${out_dir}" \
    --model Qwen/Qwen2.5-1.5B \
    --max-model-len 4096 --kv-cache-memory-bytes 134217728 \
    --prompt-repeats 260 260 260 260 260 260 260 260 \
    --prompt-style legacy_disjoint_v1 \
    --batch-prompts --batch-restore-transfers \
    --error-bound 1e-4 --cuszp-mode fixed \
    --adaptive-cuszp-modes fixed \
    --adaptive-profile data/qwen2.5-1.5b_mixed_21x1e-4_7x1e-5.json \
    --adaptive-candidates 1e-5 1e-4 \
    --pcie-service-rate-gbps 154 --transfer-deadline-ms 2 \
    --cpu-offload-gb 4 --gpu-memory-utilization 0.8 --max-tokens 8 \
    --quality-gate --quality-baseline-method async_raw \
    --quality-min-token-match-rate 0.875 --quality-min-exact-match-rate 0.875 \
    --quality-max-token-match-drop 0 --quality-max-exact-match-drop 0 \
    "${extra[@]}"
}

run_experiment_suite() {
  local root="$1"
  local resume="$2"
  local extra=()
  [[ "${resume}" == "1" ]] && extra+=(--resume)

  "${PYTHON_BIN}" benchmarks/run_gate_d_suite.py \
    --workload-version legacy_v1 --out-root "${root}/gate_d_fair_async_probe" \
    --trials 5 --require-all-models "${extra[@]}"
  "${PYTHON_BIN}" benchmarks/run_concurrency_sweep.py \
    --out-root "${root}/qwen1.5b_real_concurrency_v3" --trials 5 "${extra[@]}"
  "${PYTHON_BIN}" benchmarks/run_arrival_rate_sweep.py \
    --out-root "${root}/qwen1.5b_open_loop_arrival_v1" --trials 5 "${extra[@]}"
  "${PYTHON_BIN}" benchmarks/run_arrival_rate_sweep.py \
    --model Qwen/Qwen2.5-0.5B --rates 8 \
    --out-root "${root}/qwen0.5b_open_loop_arrival_v1" --trials 5 "${extra[@]}"
  run_quality_suite "${root}/infocom_long_context_quality_v1/five_trial" "${resume}"
  run_adaptive_pilot "${root}/vllm_qwen1.5b_4k_gate_c_packed_mixed21_complete_2trial" "${resume}"
}

dry_run() {
  verify_full
  local scratch
  scratch="$(mktemp -d)"
  trap 'rm -rf -- "$scratch"; trap - RETURN' RETURN

  "${PYTHON_BIN}" benchmarks/run_gate_d_suite.py \
    --workload-version legacy_v1 --out-root "${scratch}/gate_d" \
    --trials 5 --dry-run --require-all-models
  "${PYTHON_BIN}" benchmarks/run_concurrency_sweep.py \
    --out-root "${scratch}/concurrency" --trials 5 --dry-run
  "${PYTHON_BIN}" benchmarks/run_arrival_rate_sweep.py \
    --out-root "${scratch}/arrival" --trials 5 --dry-run
  "${PYTHON_BIN}" benchmarks/run_arrival_rate_sweep.py \
    --model Qwen/Qwen2.5-0.5B --rates 8 \
    --out-root "${scratch}/arrival-small" --trials 5 --dry-run
  run_quality_suite "${scratch}/quality" 0 1
  run_adaptive_pilot "${scratch}/adaptive" 0 1
}

experiments() {
  verify_full
  build_native
  run_experiment_suite data 1
  echo "Experiments finished. Review changed formal outputs before accepting them."
  echo "After review, refresh hashes with:"
  echo "  ${PYTHON_BIN} reproducibility/build_hash_manifest.py --write"
}

generate_assets_from_root() {
  local data_root="$1"
  local tables_out="$2"
  local figures_out="$3"
  "${PYTHON_BIN}" benchmarks/generate_paper_assets.py \
    --data-root "${data_root}" --out-dir "${tables_out}"
  "${PYTHON_BIN}" benchmarks/generate_infocom_figures.py \
    --qwen1p5-arrival "${data_root}/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json" \
    --qwen0p5-arrival "${data_root}/qwen0.5b_open_loop_arrival_v1/arrival_rate_summary.json" \
    --gate-d "${data_root}/gate_d_fair_async_probe/gate_d_summary.json" \
    --out-dir "${figures_out}"
}

rerun() {
  if [[ -z "${REPRO_OUT_ROOT:-}" ]]; then
    echo "rerun requires REPRO_OUT_ROOT=data/reproductions/<new-run>" >&2
    exit 2
  fi
  local resolved allowed_root
  resolved="$(realpath -m "${WORKSPACE_ROOT}/${REPRO_OUT_ROOT}")"
  allowed_root="${WORKSPACE_ROOT}/data/reproductions/"
  if [[ "${resolved}/" != "${allowed_root}"* ]]; then
    echo "REPRO_OUT_ROOT must be a new directory under data/reproductions/" >&2
    exit 2
  fi
  if [[ -d "${resolved}" ]]; then
    if [[ -n "$(find "${resolved}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
      echo "Independent rerun directory is not empty: ${resolved}" >&2
      exit 2
    fi
  fi
  verify_full
  build_native
  run_experiment_suite "${resolved}" 0
  generate_assets_from_root "${resolved}" "${resolved}/paper_generated" "${resolved}/figures"
  echo "Independent rerun completed without reusing canonical trials: ${resolved}"
}

assets() {
  "${PYTHON_BIN}" benchmarks/generate_paper_assets.py --sync-paper
  "${PYTHON_BIN}" benchmarks/generate_infocom_figures.py
}

verify_paper() {
  verify_full
  "${PYTHON_BIN}" benchmarks/generate_paper_assets.py --check-paper
  "${PYTHON_BIN}" -m pytest -q \
    tests/test_generate_paper_assets.py \
    tests/test_generate_infocom_figures.py \
    tests/test_reproducibility_contract.py
}

main() {
  require_python
  cd "${WORKSPACE_ROOT}"
  case "${1:-validate}" in
    validate) validate ;;
    dry-run) dry_run ;;
    calibrate) calibrate_adaptive 0 ;;
    calibrate-fresh) calibrate_adaptive 1 ;;
    experiments) experiments ;;
    rerun) rerun ;;
    assets) assets ;;
    verify-paper) verify_paper ;;
    full) validate; calibrate_adaptive 0; experiments; assets ;;
    -h|--help|help) usage ;;
    *) usage >&2; exit 2 ;;
  esac
}

main "$@"
