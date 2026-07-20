# Reducing PCIe Bottlenecks for LLM Serving with cuSZp-Compressed KV-Cache Transfer

This repository provides a research prototype for reducing CPU-GPU KV-cache transfer overhead in LLM serving systems. The core idea is to compress KV-cache blocks before swap-in/swap-out so that the cost of PCIe transfer is reduced while preserving acceptable reconstruction quality.

The implementation contains:

- a cuSZp-based compression wrapper in [integration/cuszp_wrapper](integration/cuszp_wrapper)
- a real vLLM 0.23 compressed offload connector with a deadline-pressure
  adaptive controller in
  [integration/compression_pipeline/vllm_v1_compressed_offload.py](integration/compression_pipeline/vllm_v1_compressed_offload.py)
- benchmark and evaluation scripts that measure transfer time, effective bandwidth, compression ratio, and reconstruction error

This project is best understood as an experimental research artifact and prototype integration layer rather than a production-ready vLLM fork.

> **Result provenance:** obsolete simulation-only pipelines and figures have
> been removed. Paper claims must come from the real vLLM JSONL/aggregate files
> under `data/`; missing experiments remain missing rather than using fallbacks.

## 1. Reproduction overview

The repository is designed to reproduce the main experimental story of the paper:

1. compress KV-cache blocks before CPU-GPU transfer,
2. evaluate the resulting swap-in/swap-out time and effective bandwidth,
3. compare the proposed method against baseline, INT8, and zlib-based baselines,
4. generate figures that highlight the transfer-time benefit.

## 2. Code file guide

This section summarizes the purpose of the main code files in the repository.

### Entry points and orchestration

- [run.sh](run.sh): builds the native wrapper when requested, runs the maintained test suite, and optionally launches a real vLLM smoke experiment.
- [test.sh](test.sh): wrapper for the maintained validation workflow.

### Benchmark and evaluation scripts

- [benchmarks/smoke_vllm_compressed_offload.py](benchmarks/smoke_vllm_compressed_offload.py): executes the real vLLM offload path and records request, transfer, quality, and restore-stage metrics.
- [benchmarks/run_vllm_repeated_smoke.py](benchmarks/run_vllm_repeated_smoke.py): runs isolated warmed vLLM trials with Student-t 95% confidence intervals.
- [benchmarks/layer_sensitivity_sweep.py](benchmarks/layer_sensitivity_sweep.py): calibrates model-specific per-layer error safety caps.
- [benchmarks/build_joint_sensitivity_profile.py](benchmarks/build_joint_sensitivity_profile.py): combines sensitivity evidence into a connector profile.
- [benchmarks/benchmark_restore_break_even.py](benchmarks/benchmark_restore_break_even.py): reports the measured raw-versus-compressed restore break-even.
- [benchmarks/probe_cuszp_sizes.py](benchmarks/probe_cuszp_sizes.py): checks cuSZp correctness and payload size across input sizes and encoding modes.
- [benchmarks/plot_vllm_repeated_e2e.py](benchmarks/plot_vllm_repeated_e2e.py): plots measured repeated vLLM aggregates without simulated fallback.

### Compression and integration logic

- [integration/compression_pipeline/vllm_v1_compressed_offload.py](integration/compression_pipeline/vllm_v1_compressed_offload.py): real vLLM connector, compressed host store, batched restore, and Gate C controller.
- [integration/compression_pipeline/native_lossless.py](integration/compression_pipeline/native_lossless.py): zstd and LZ4 comparison codecs.
- [integration/cuszp_wrapper/cuszp_wrapper.cpp](integration/cuszp_wrapper/cuszp_wrapper.cpp): C++ cuSZp wrapper with per-call error bounds and CUDA stream support.
- [integration/cuszp_wrapper/pybind11_bindings.cpp](integration/cuszp_wrapper/pybind11_bindings.cpp): Python bindings.
- [integration/cuszp_wrapper/cuszp_wrapper.h](integration/cuszp_wrapper/cuszp_wrapper.h): wrapper interface.

### Tests

- [tests/test_vllm_v1_compressed_offload.py](tests/test_vllm_v1_compressed_offload.py): exercises raw fallback, all cuSZp modes, adaptive bounds, joint mode choice, batching, and restore.
- [tests/test_layer_sensitivity.py](tests/test_layer_sensitivity.py): validates sensitivity classification and profile construction.
- [tests/test_native_lossless.py](tests/test_native_lossless.py): checks lossless comparison codecs.
- [tests/test_repeated_smoke.py](tests/test_repeated_smoke.py): validates repeated-trial aggregation.

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
cmake -S integration/cuszp_wrapper -B integration/cuszp_wrapper/build_local \
  -Dpybind11_DIR=$(python -m pybind11 --cmakedir) \
  -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
cmake --build integration/cuszp_wrapper/build_local -j$(nproc)
```

Build failures stop the workflow; paper experiments never silently replace cuSZp with zlib.

## 4. Running the full workflow

The recommended entry point is:

```bash
./test.sh
```

By default this runs the maintained test suite against the existing native extension.
Set `SKIP_BUILD=0` to rebuild the wrapper first. Set `RUN_VLLM_SMOKE=1` to
also launch a real vLLM smoke experiment; `MODEL`, `CUSZP_MODE`,
`ERROR_BOUND`, `METRICS`, and `SUMMARY` can be overridden as environment
variables.

The workflow has no synthetic benchmark path and no silent codec fallback.
Layer sensitivity, repeated comparisons, and paper figures are explicit
experiments because their model and workload settings must be recorded.

## 5. Running individual experiments

### cuSZp encoding-mode probe

```bash
./venv/bin/python benchmarks/probe_cuszp_sizes.py \
  --cuszp-mode fixed --error-bound 1e-5 --sizes 36864 65536
```

### Layer sensitivity sweep

```bash
PYTHONPATH=integration/compression_pipeline ./venv/bin/python3.12 benchmarks/layer_sensitivity_sweep.py \
  --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2
```

### Measured figure generation

```bash
./venv/bin/python benchmarks/plot_vllm_repeated_e2e.py \
  --input data/vllm_qwen1.5b_4k_concurrency8_batched_common_codecs_5trial/aggregate.json \
  --output data/figures/vllm_qwen1.5b_4k_comparison.png
```

### Real vLLM v1 offload smoke test

```bash
./venv/bin/python benchmarks/smoke_vllm_compressed_offload.py \
  --model Qwen/Qwen2.5-0.5B \
  --metrics data/vllm_offload_smoke.jsonl \
  --error-bound 1e-4
```

This uses vLLM 0.23's `OffloadingConnector` and the project-local
`CompressedCPUOffloadingSpec`; it does not patch the installed vLLM package.
The script succeeds only after observing a real GPU-to-CPU offload event.

On the RTX 5080 development machine, both GPU-to-CPU and CPU-to-GPU events
have been verified. The vLLM canonical cache is an `int8` byte view of BF16
storage, so the connector must reinterpret those bytes as BF16 before any
numerical codec. Results produced before that fix are invalid and are listed
in `data/EXPERIMENT_PROVENANCE.md`.

After the byte-view fix, the Qwen2.5-0.5B smoke workload measured about
`2.054x` transfer compression for static cuSZp at `1e-4` and `1.9999x` for
per-page symmetric INT8. In a corrected six-request initial/replay check, raw
offload matched 6/6 sequences, cuSZp matched 5/6 (83.33% mean token agreement),
and INT8 matched 4/6 (81.25% mean token agreement). See
`data/QUALITY_SMOKE_REPORT.md`. These are single smoke runs, not paper results;
formal quality, repeated trials, end-to-end latency, and broader workloads are
still required before claiming an advantage.

The canonical common-codec comparison is
`data/vllm_common_codecs_5trial/aggregate.json`. On this workload, asynchronous
cuSZp reduces transfer bytes by about 2.054x, versus 1.239x for lossless zstd
and 1.000x for LZ4 after raw fallback. Initial E2E is 61.24 ms for stock,
75.10 ms for zstd, 74.33 ms for LZ4, and 82.60 ms for cuSZp. Thus cuSZp's
current demonstrated advantage is byte reduction, not small-workload latency.
The corresponding five-trial long-context check is
`data/vllm_2k_common_codecs_5trial/aggregate.json`. At about 1.7K prompt
tokens, quality-matched cuSZp `1e-5` preserves 6/6 sequences and reduces bytes
by 1.441x, compared with 1.239x for zstd and 1.000x for LZ4. Initial E2E is
82.88 ms for stock, 168.31 ms for cuSZp, 136.84 ms for zstd, and 129.54 ms
for LZ4. Longer context alone therefore does not yet produce a latency win;
the compressed-store implementation remains the primary bottleneck.
The adaptive controller has also been observed changing
GREEN→YELLOW→RED on real variable-size vLLM jobs while preserving all six
replayed sequences. These results validate the data path and controller
mechanism, not an end-to-end speedup claim.

The current repeated concurrent pressure comparison is stored in
`data/vllm_adaptive_vs_static_5trial/aggregate_optimized_adaptive.json`.
Adaptive changes
GREEN→YELLOW→RED in every trial and preserves all six sequences, but is
currently dominated by quality-matched static cuSZp `1e-5` in both compression
ratio (1.304x versus 1.445x) and initial E2E (208.75 ms versus 183.79 ms).
CUDA layer-index caching and the GREEN whole-page fast path reduce adaptive
initial E2E by 10.9%, but do not yet reverse that ordering.
This is a confirmed optimization target, not a positive adaptive-performance
claim.

The vLLM connector's adaptive mode requires a calibrated layer sensitivity
profile, a measured uncompressed PCIe service rate, and a transfer deadline.
It models outstanding bytes at that measured rate, changes GREEN/YELLOW/RED
state with hysteresis, and never selects a per-layer bound above the offline
safety cap. Every GPU-to-CPU event records the pressure, estimated backlog,
state, transition flag, and selected bound counts. The existing sensitivity
profiles are debugging artifacts and must be recalibrated before this mode is
used for paper results.

## 6. Expected artifacts

Real experiments write explicit, model-scoped artifacts under `data/`:

- per-trial JSONL metrics, summaries, and logs in a named experiment directory,
- `aggregate.json` with repeated-trial means and 95% confidence intervals,
- model-specific layer and joint sensitivity profiles,
- hardware/model-specific Gate C mode calibration profiles, and
- measured-only PNG/PDF figures under `data/figures/`.

## 7. Notes

- The benchmark scripts download Hugging Face models and run GPU-based measurements, so runtime can be long.
- The optional zstd and LZ4 baselines lazily load the system `libzstd.so.1` and
  `liblz4.so.1` libraries. cuSZp, raw, INT8, and zlib modes do not require them.
- The current implementation is a prototype intended for experimental validation rather than direct production deployment in a shipping vLLM stack.
- Some workflows are sensitive to the exact CUDA, PyTorch, and cuSZp versions available in the environment.

## 8. RTX 5080 batched restore and cost-aware mode

The connector can profile CPU decode, H2D, GPU decode, and scatter separately
with `--profile-restore-stages`. The optional
`--batch-restore-transfers` path queues pinned payload copies together and
synchronizes once before decode. On Qwen2.5-1.5B, 4K context, and eight
concurrent requests, batching reduced mean CPU-to-GPU handler time from
16.964 to 10.848 ms for raw pages and from 43.916 to 31.907 ms for cuSZp.

The corrected weighted throughput is 157.17 Gbit/s for raw H2D, 154.12 Gbit/s
for compressed H2D, and only 24.65 Gbit/s for cuSZp decompression. cuSZp
reduces restore traffic from 64.82 to 41.48 MB per event, but its 21.03 ms
decode cost makes it slower than raw on this RTX 5080 workload. This repository
therefore does not claim a universal cuSZp latency win.

Adaptive mode can additionally enable `--cost-aware-restore` with calibrated
compression ratios, H2D bandwidth, decompression throughput, fixed overhead,
and a minimum predicted saving. A real two-trial Qwen2.5-0.5B run reached
GREEN/YELLOW/RED pressure states while the cost gate correctly selected raw
for every page. This is a no-regret fallback result, not an adaptive speedup.

Use `--aggregate-only` with
[benchmarks/run_vllm_repeated_smoke.py](benchmarks/run_vllm_repeated_smoke.py)
to recompute an aggregate from existing JSONL and summary files without
rerunning GPU trials. H2D and decompression throughput are aggregated as total
bytes divided by total time.

Detailed commands, results, break-even analysis, and claim boundaries are in
[data/RTX5080_BATCHED_RESTORE_REPORT.md](data/RTX5080_BATCHED_RESTORE_REPORT.md).

## 9. Gate C: joint cuSZp policy

Gate C does not introduce a new codec. It jointly chooses:

1. raw transfer or cuSZp,
2. a model-safe per-layer error bound, and
3. cuSZp's existing `plain`, `fixed`, or `outlier` encoding mode.

The controller first applies the offline layer safety caps, then uses measured
compression ratio and decompression throughput for each mode to minimize
predicted restore time. If no calibrated compressed plan clears the configured
minimum saving, the affected layers use raw transfer.

On the preliminary Qwen2.5-1.5B, 4K, concurrency-8 calibration, `fixed` at
`1e-5` reached 1.615x compression and 29.51 ms profiled restore, versus
1.563x and 36.27 ms for `plain`, and 1.563x and 41.14 ms for `outlier`.
The two-trial `fixed` results are calibration evidence, not final confidence
intervals. Uniform `1e-4` and `1e-3` substantially damaged token agreement, so
those bounds may only be used where a model-specific sensitivity profile marks
individual layers safe.

The current RTX 5080 calibration is
[data/gate_c_qwen1.5b_rtx5080_mode_profile.json](data/gate_c_qwen1.5b_rtx5080_mode_profile.json).
A sensitivity profile must match the exact model; the existing 0.5B profiles
must not be reused for Qwen2.5-1.5B or for any of the other planned models.

Implementation details, calibration tables, actual-link results, and remaining
INFOCOM evidence are in [data/GATE_C_REPORT.md](data/GATE_C_REPORT.md).

## 10. Fused fixed-cuSZp BF16 restore

The fixed-mode restore path now preserves the original cuSZp bitstream and
error-bound rule while changing how it executes:

- decompression metadata uses persistent grow-only GPU workspaces;
- pages with one shape and mode are decoded by a single batched CUDA launch;
- fixed-mode output is written directly to BF16 rather than materializing an
  intermediate FP32 batch; and
- BF16 output uses vectorized two-element stores.

On CUDA 12.0 and RTX 5080, build with PTX enabled:

```bash
TORCH_CUDA_ARCH_LIST="9.0+PTX" SKIP_BUILD=0 ./run.sh
```

The maintained workflow sets this value by default. The rebuilt extension and
the fixed-BF16 round-trip path pass all 27 tests.

For Qwen2.5-1.5B, 4K context, concurrency eight, fixed 1e-5, and batched
restore, the five-trial profiled means are:

| Metric | Raw | cuSZp |
|---|---:|---:|
| Restore payload ratio | 1.000x | 1.614x |
| CPU-to-GPU handler | 9.125 ms | 8.882 ms |
| Profiled restore total | 8.481 ms | 8.016 ms |
| H2D stage | 2.958 ms | 1.991 ms |
| GPU decode | 0 ms | 0.554 ms |
| Replay E2E | 335.664 ms | 335.878 ms |

This passes the mean restore break-even but not the end-to-end paper gate.
The restore confidence intervals overlap, replay E2E is tied, and initial
compression/store preprocessing is still slower than raw. At concurrency
sixteen, finer restore fragmentation makes cuSZp slightly slower than raw.
Results must therefore be reported by actual restore batch shape, not by
nominal concurrency alone.

The historical aggregates above used grouped method execution. New repeated
experiments default to interleaved order, such as raw trial 1, cuSZp trial 1,
raw trial 2, cuSZp trial 2. Use `--trial-order grouped` only to reproduce old
experiments. Final paper tables should use the interleaved default.

## 11. Current fair-direct and PCIe-contention result

Section 10 is retained as historical provenance. The current implementation
packs all compressed host payloads into one pinned slab, issues one H2D copy,
and lets the fixed BF16 cuSZp kernel write directly into final KV pages. The
raw baseline likewise copies pinned host bytes directly into final KV pages.
Neither path uses a GPU staging/scatter pass.

The previous Python-side current-stream synchronization was redundant and has
been removed. The maintained test suite now passes 29 tests.

For Qwen2.5-1.5B, 4K context, eight concurrent disjoint prompts, fixed cuSZp
`1e-5`, and five interleaved trials:

| Operating point | Paired metric (cuSZp - raw) | 95% CI half-width | Result |
|---|---:|---:|---|
| No contention, profiled completion | restore total -0.133 ms | 0.057 ms | cuSZp faster |
| No contention, normal handler | CPU-to-GPU -0.455 ms | 0.346 ms | cuSZp faster, 5/5 |
| Medium contender, profiled completion | restore total -0.461 ms | 0.149 ms | cuSZp faster, 5/5 |
| Medium contender, profiled handler | CPU-to-GPU +0.038 ms | 0.157 ms | tied |
| High contender, profiled completion | restore total -0.950 ms | 0.293 ms | cuSZp faster, 5/5 |
| High contender, profiled handler | CPU-to-GPU -0.466 ms | 0.290 ms | cuSZp faster, 5/5 |
| High contender | replay E2E -70.084 ms | 24.777 ms | cuSZp faster, 5/5 |

The high contender transfers a measured 405.9 Gbit/s for raw trials and
415.0 Gbit/s for cuSZp trials. The medium configuration inserts a 1500 us idle
interval after each 64 MiB contender copy and measures 164.0 and 183.8 Gbit/s.
Use `--pcie-contender-mib` and `--pcie-contender-idle-us` to reproduce
controlled break-even points.

Quality is unchanged: raw and cuSZp both reproduce 7/8 requests. Replay E2E is
statistically tied without contention. Initial E2E is still worse for cuSZp
without contention (paired +184.114 +/- 86.026 ms), so the GPU-to-CPU
compression/store path is the next optimization target.

Canonical aggregates:

- `data/vllm_qwen1.5b_4k_concurrency8_fair_direct_profile_interleaved_5trial/aggregate.json`
- `data/vllm_qwen1.5b_4k_concurrency8_fair_direct_nosync_noprofile_interleaved_5trial/aggregate.json`
- `data/vllm_qwen1.5b_4k_concurrency8_pcie_contender64_idle1500_profile_interleaved_5trial/aggregate.json`
- `data/vllm_qwen1.5b_4k_concurrency8_pcie_contender64_profile_interleaved_5trial/aggregate.json`

## 12. Direct pinned store and persistent compression workspace

The GPU-to-CPU production path no longer materializes each payload in pageable
CPU memory, pins it with a second copy, and copies it again into the packed
slab. It retains GPU payloads until the job is complete, copies every segment
directly into its final pinned-slab slice, and synchronizes the D2H batch once.
The fixed cuSZp compressor also reuses the wrapper's grow-only offset/flag
workspace instead of performing three cudaMalloc/cudaFree pairs per page.

Two-trial Qwen2.5-1.5B probes show:

| Version | Raw GPU-to-CPU | cuSZp GPU-to-CPU | Raw initial E2E | cuSZp initial E2E |
|---|---:|---:|---:|---:|
| Direct pinned slab | 98.658 ms | 138.665 ms | 717.155 ms | 1195.587 ms |
| Plus persistent fixed workspace | 100.174 ms | 127.914 ms | 744.753 ms | 1112.142 ms |

Both probes preserve the same 0.875 token/exact match, and cuSZp CPU-to-GPU
remains faster in both trials. These are optimization probes, not final
confidence intervals. They supersede the implementation used by the earlier
five-trial initial-E2E table; final tables must be rerun after batched
compression removes per-page range and compressed-length synchronization.

Aggregate: `data/vllm_qwen1.5b_4k_concurrency8_persistent_compress_workspace_noprofile_probe_2trial/aggregate.json`.

## 13. BF16 single-grid compression crosses the fair raw baseline

Section 12 is retained as optimization provenance. The production fixed-mode
path now reads native BF16 KV pages directly, computes every page's relative
range with one batched CUDA reduction, and compresses all same-shaped pages in
one cuSZp-compatible grid. It preserves the fixed-mode byte layout and the
per-page actual absolute error bound. Compressed outputs reuse a grow-only GPU
slab, and sizes plus actual bounds return in one stream synchronization.

The repeated runner now exposes `async_raw`. End-to-end comparisons must pair
`async_raw` with `async_cuszp_1e-5`; mixing synchronous raw with asynchronous
cuSZp is not a valid scheduling comparison. The maintained suite passes 38
tests, including an assertion that the production path actually invokes the
BF16 batch API.

For Qwen2.5-1.5B, 4K context, eight concurrent disjoint prompts, fixed cuSZp
`1e-5`, and five interleaved trials without synthetic contention:

| Metric | async raw | async cuSZp | Paired cuSZp - raw | 95% CI half-width |
|---|---:|---:|---:|---:|
| GPU-to-CPU handler | 72.892 ms | 46.570 ms | -26.322 ms | 8.631 ms |
| CPU-to-GPU handler | 1.565 ms | 0.970 ms | -0.596 ms | 0.141 ms |
| Initial E2E | 932.269 ms | 743.984 ms | -188.285 ms | 22.608 ms |
| Replay E2E | 327.121 ms | 305.393 ms | -21.728 ms | 30.908 ms |

GPU-to-CPU, CPU-to-GPU, and initial E2E win in all five pairs; their confidence
intervals exclude zero. Replay E2E has a favorable mean but is not statistically
significant. Compression ratio is 1.61445x, and both methods retain 0.875 token
match and 0.875 exact match. This passes the static full-datapath prerequisite
for Gate C. The adaptive Gate C claim remains open until a cost-aware policy
beats both always-raw and this improved always-cuSZp baseline.

Canonical aggregate:
`data/vllm_qwen1.5b_4k_concurrency8_async_fair_bf16_single_grid_reduction_5trial/aggregate.json`.

## 14. Adaptive layer segments use indexed cuSZp kernels

The adaptive fixed path no longer materializes BF16 `index_select` tensors.
New cuSZp-compatible indexed compression and decompression kernels map the
logical segment index directly to the original `[K/V, layer, ...]` page. The
fixed bitstream layout and relative error-bound calculation are unchanged.
Unsupported layouts retain the gathered batch and per-segment fallbacks.

Two RED-state engineering probes on the Qwen2.5-1.5B 4K workload reduced the
adaptive GPU-to-CPU handler from 83.231 ms to 54.219 ms. Indexed scatter reduced
CPU-to-GPU from 4.151 ms to 2.842 ms. Compression ratio was 2.81565x. These are
two-trial implementation probes, not paper evidence.

The existing joint sensitivity file is only a debug profile: it enables 1e-3
for 27/28 layers and produced 0.28125 token match and 0.25 exact match in the
latest probe, versus 0.875 for the quality-matched raw/static runs. Adaptive
Gate C therefore remains open. The next hard gate is multi-prompt sensitivity
calibration with an automatic full-workload quality rejection rule.

Canonical engineering probe:
`data/vllm_qwen1.5b_4k_concurrency8_adaptive_indexed_bidir_red_probe_2trial/aggregate.json`.

## 15. Multi-prompt quality gate and phased adaptive trace

NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
NaN

NaN
NaN
NaN
NaN
