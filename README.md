# Reducing PCIe Bottlenecks for LLM Serving with cuSZp-Compressed KV-Cache Transfer

This repository provides a research prototype for reducing CPU-GPU KV-cache transfer overhead in LLM serving systems. The core idea is to compress KV-cache blocks before swap-in/swap-out so that the cost of PCIe transfer is reduced while preserving acceptable reconstruction quality.

The implementation contains:

- a cuSZp-based compression wrapper in [integration/cuszp_wrapper](integration/cuszp_wrapper)
- a vLLM-style compressed swap path in [integration/compression_pipeline/compressed_swap.py](integration/compression_pipeline/compressed_swap.py)
- a congestion-aware scheduler in [integration/compression_pipeline/adaptive_scheduler.py](integration/compression_pipeline/adaptive_scheduler.py)
- benchmark and evaluation scripts that measure transfer time, effective bandwidth, compression ratio, and reconstruction error

This project is best understood as an experimental research artifact and prototype integration layer rather than a production-ready vLLM fork.

## 1. Reproduction overview

The repository is designed to reproduce the main experimental story of the paper:

1. compress KV-cache blocks before CPU-GPU transfer,
2. evaluate the resulting swap-in/swap-out time and effective bandwidth,
3. compare the proposed method against baseline, INT8, and zlib-based baselines,
4. generate figures that highlight the transfer-time benefit.

## 2. Code file guide

This section summarizes the purpose of the main code files in the repository.

### Entry points and orchestration

- [run.sh](run.sh): main end-to-end workflow entry script that builds the runtime, runs benchmarks, evaluates policies, and generates figures.
- [test.sh](test.sh): simple wrapper for the default workflow entry point.

### Benchmark and evaluation scripts

- [benchmarks/benchmark_pipeline.py](benchmarks/benchmark_pipeline.py): measures baseline GPU-CPU transfer bandwidth and cuSZp compression/decompression cost for KV-cache tensors.
- [benchmarks/evaluate_policies.py](benchmarks/evaluate_policies.py): compares multiple policies including baseline, static cuSZp, adaptive cuSZp, INT8, and zlib.
- [benchmarks/layer_sensitivity_sweep.py](benchmarks/layer_sensitivity_sweep.py): evaluates how compressing different layers affects reconstruction quality and downstream model behavior.
- [benchmarks/plot_summary.py](benchmarks/plot_summary.py): generates the main summary plots for transfer time and bandwidth comparisons.
- [benchmarks/run_pareto_queue_ablation.py](benchmarks/run_pareto_queue_ablation.py): generates additional paper-style figures such as Pareto boundary and queue-depth ablation plots.

### Compression and integration logic

- [integration/compression_pipeline/compressed_swap.py](integration/compression_pipeline/compressed_swap.py): implements the compressed swap path and monkey-patching logic for vLLM-style cache engine swapping.
- [integration/compression_pipeline/adaptive_scheduler.py](integration/compression_pipeline/adaptive_scheduler.py): implements the congestion-aware scheduler that assigns different error bounds according to swap pressure.
- [integration/compression_pipeline/offloading_wrapper.py](integration/compression_pipeline/offloading_wrapper.py): provides wrapper logic for offloading-related hooks and transfer handling.
- [integration/cuszp_wrapper/cuszp_wrapper.cpp](integration/cuszp_wrapper/cuszp_wrapper.cpp): C++ implementation of the cuSZp wrapper used by the Python bindings.
- [integration/cuszp_wrapper/pybind11_bindings.cpp](integration/cuszp_wrapper/pybind11_bindings.cpp): exposes the cuSZp C++ functionality to Python via PyBind11.
- [integration/cuszp_wrapper/cuszp_wrapper.h](integration/cuszp_wrapper/cuszp_wrapper.h): header definitions for the wrapper interface.

### Tests

- [tests/test_evaluate_policies.py](tests/test_evaluate_policies.py): validates that the evaluation pipeline still works when cuSZp is unavailable.
- [tests/test_plot_summary.py](tests/test_plot_summary.py): ensures that the summary plotting workflow produces the expected figure output.

## 3. Environment requirements

A Linux machine with:

- Python 3.12
- CUDA-capable NVIDIA GPU
- CUDA 12.x toolchain
- optional: a local cuSZp installation under /opt/cuSZp
- network access for Hugging Face model downloads

Create and activate a virtual environment in the repository root:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## 3. Build the cuSZp Python extension (optional but recommended)

If you want to use the native cuSZp path, build the Python extension from the repository root:

```bash
cd integration/cuszp_wrapper
rm -rf build_local && mkdir -p build_local && cd build_local
cmake .. \
  -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) \
  -DCMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)") \
  -DCMAKE_CXX_FLAGS="$(python3 -c 'import torch.utils.cpp_extension as E; print(" ".join("-I"+p for p in E.include_paths()))')"
cmake --build . -j$(nproc)
```

If the build fails, the workflow will still continue in a CPU/zlib fallback mode.

## 4. Running the full workflow

The recommended entry point is:

```bash
./test.sh
```

This script supports two modes:

- with a local cuSZp installation, it uses the real cuSZp compressor for CPU-GPU KV-cache transfer experiments;
- without cuSZp, it automatically falls back to CPU zlib compression so the workflow still produces benchmark outputs and figures.

The workflow runs the following steps:

1. prepare the compression runtime,
2. run the unified benchmark pipeline,
3. profile layer sensitivity,
4. evaluate compression policies,
5. generate summary figures.

## 5. Running individual experiments

### Policy evaluation

```bash
PYTHONPATH=integration/compression_pipeline ./venv/bin/python3.12 benchmarks/evaluate_policies.py \
  --models gpt2 --out data/eval_summary.json --iterations 10
```

### Layer sensitivity sweep

```bash
PYTHONPATH=integration/compression_pipeline ./venv/bin/python3.12 benchmarks/layer_sensitivity_sweep.py \
  --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2
```

### Figure generation

```bash
./venv/bin/python3.12 benchmarks/run_pareto_queue_ablation.py
./venv/bin/python3.12 benchmarks/plot_summary.py
```

## 6. Expected artifacts

Running the workflow produces the following artifacts under data/:

- data/benchmark_summary.md
- data/eval_summary.json
- data/layer_sensitivity.json
- data/figures/summary_cpu_gpu_transfer_comparison.png
- per-model figures under data/figures/
- model-specific KV-cache snapshots such as data/gpt2_kv_cache.pt

## 7. Notes

- The benchmark scripts download Hugging Face models and run GPU-based measurements, so runtime can be long.
- The current implementation is a prototype intended for experimental validation rather than direct production deployment in a shipping vLLM stack.
- Some workflows are sensitive to the exact CUDA, PyTorch, and cuSZp versions available in the environment.
