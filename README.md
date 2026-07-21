# cuSZp KV-cache transfer for long-context LLM serving

This repository is the implementation and artifact for an INFOCOM-oriented
study of CPU--GPU KV-cache movement. It integrates the existing cuSZp codec
into vLLM's offloading path and asks a systems question:

> When does reducing PCIe bytes outweigh compression, decompression, packing,
> and scheduling overhead?

The project does not claim that compression is always faster. Its current
evidence shows conditional raw break-even on the tested RTX 5080 platform,
positive results under selected Qwen2.5 concurrent/open-loop workloads, and
explicit negative results for models, codecs, and adaptive policies that do
not pass latency or quality gates.

## What is implemented

- A vLLM 0.23 `OffloadingConnector` integration that intercepts the real
  KV-cache data path without modifying the installed vLLM package.
- C++/CUDA/pybind11 bindings around pinned upstream cuSZp, with explicit
  worker-stream ordering and batched fixed-mode kernels.
- Event-ordered background store, pinned-host transfer, batched restore,
  metadata management, raw fallback, and measured stage-level timing.
- Fixed and model-calibrated error bounds. Adaptive mode chooses only among
  cuSZp configurations and raw fallback; it is not a new codec.
- Fair raw/cuSZp/INT8/zstd/LZ4 comparison protocols, distinct-request
  concurrency and open-loop arrival workloads, quality gates, and paired
  repeated trials.
- Deterministic generation of the manuscript's tables and figures from
  canonical measured JSON, with exact checks for the six inline result blocks.

## Canonical paper source

The current manuscript source is:

- `paper/conference_101719.tex`

It uses an inline IEEE `thebibliography`; no BibTeX file is required. The
older final-year-project source may be retained locally under `paper/archive/`
as an idea reference. Its `.tex` and `.bib` files are Git-ignored and are not
part of the INFOCOM build or evidence chain.

No repository command compiles the PDF. For Overleaf, upload only
`paper/conference_101719.tex`, `IEEEtran.cls`, and the three referenced PNG
figures. The result tables and IEEE bibliography are already inline.

## Reproducibility contract

The reviewed platform and exact dependencies are frozen in
`reproducibility/`:

- Python 3.12.3 and exact package versions in `requirements.txt`
- cuSZp commit `f581dcf329c907c320f4743a9c6e7ee2fb9c5494`
- eight exact Hugging Face model snapshot revisions
- RTX 5080, driver, CUDA, compiler, and CPU information
- a SHA-256 manifest for every formal JSON/JSONL/provenance artifact,
  including the adaptive prompt suite and eight per-prompt calibration profiles

Data are ignored by default. Only paths declared in
`reproducibility/formal_artifacts.json` are admitted as paper evidence.
Historical, exploratory, and simulated outputs elsewhere under `data/` are
not used by `conference_101719.tex`.

## Setup

Create the exact Python environment:

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

The reviewed machine has the pinned cuSZp checkout at `/opt/cuSZp`. To
install it in a user-writable location instead:

```bash
CUSZP_ROOT="$PWD/.deps/cuSZp" reproducibility/install_cuszp.sh
```

Build the project wrapper by setting `SKIP_BUILD=0`:

```bash
CUSZP_ROOT="$PWD/.deps/cuSZp" SKIP_BUILD=0 ./run.sh validate
```

The build never silently substitutes zlib or another codec for cuSZp.

## Single entry point

```bash
./run.sh validate
```

This default target verifies pinned software and formal hashes, optionally
builds the native wrapper, and runs the maintained test suite. It does not
launch model inference.

Other explicit targets are:

```bash
./run.sh dry-run          # validate every formal top-level command
./run.sh calibrate          # rebuild adaptive profile from admitted evidence
./run.sh calibrate-fresh    # rerun all eight GPU sensitivity sweeps
./run.sh experiments       # resume canonical GPU experiments; many hours
./run.sh assets            # regenerate assets and sync inline paper results
./run.sh verify-paper      # verify hashes plus exact paper/data consistency
REPRO_OUT_ROOT=data/reproductions/run-01 ./run.sh rerun
./run.sh full              # validate, calibrate, resume experiments, assets
```

Hugging Face is offline by default to prevent silent model drift. Set
`HF_HUB_OFFLINE=0` only when intentionally populating the pinned snapshots.

## Formal protocols

### Gate D: eight-model comparison

`benchmarks/run_gate_d_suite.py` runs five interleaved trials of:

- asynchronous raw
- asynchronous cuSZp at (10^{-5})
- asynchronous INT8
- asynchronous zstd
- asynchronous LZ4

The published snapshot used a historical six-seed/eight-slot workload. It is
now explicitly versioned as `legacy_v1` and reproduced by
`legacy_disjoint_v1`. `corrected_v2` creates eight unique streams and is
available for follow-up work; it is never substituted silently.

### Concurrency and open-loop arrival

`benchmarks/run_concurrency_sweep.py` uses distinct long-context requests at
concurrency 2, 4, 8, and 16. `benchmarks/run_arrival_rate_sweep.py` injects
requests at fixed wall-clock arrival times, so queuing contributes to TTFT and
end-to-end latency.

These two protocols provide the main raw-versus-cuSZp break-even evidence.
They do not use a synthetic PCIe contender.

### Quality comparison

The long-context quality suite compares raw, cuSZp, zstd, LZ4, and INT8 on
eight controlled retrieval prompts. A rejected codec remains in the artifact
as a negative result; the runner returns status 2 for a completed experiment
whose quality gate rejects at least one method.

### Adaptive calibration and pilot

`./run.sh calibrate` deterministically rebuilds the merged and selected
Qwen2.5-1.5B profile from the admitted prompt suite and eight per-prompt
teacher-forced profiles. `./run.sh calibrate-fresh` reruns those GPU sweeps.
The selected profile contains 21 layers at (10^{-4}) and seven at
(10^{-5}). The formal two-trial pilot exercises the pressure controller with
fixed cuSZp mode and `cost_aware_restore=false`; it is mechanism evidence, not
a statistical performance claim.

## Repository map

- `integration/compression_pipeline/`: vLLM connector and codec policies
- `integration/cuszp_wrapper/`: C++/CUDA/pybind11 wrapper
- `benchmarks/`: real experiment orchestrators and asset generators
- `tests/`: unit, orchestration, and paper/data consistency checks
- `reproducibility/`: environment, model, protocol, and hash manifests
- `data/`: formal and ignored exploratory measurements
- `figures/infocom/`: generated paper figures and source CSV/LaTeX data
- `paper/`: current IEEE manuscript, generated tables, and archived old source

## Claim boundaries

- All reported performance numbers come from real vLLM execution on one
  RTX 5080 / Core Ultra 7 265KF WSL2 system.
- The paper does not generalize break-even to every GPU, PCIe generation,
  model, context length, or serving engine.
- Compression ratio alone is not treated as a latency result.
- The eight-model layer sensitivity profile is not shared; error caps are
  model specific. The eight-model comparison itself uses the same fixed
  quality-safe cuSZp bound for fairness.
- The two-trial adaptive result is retained as mechanism evidence and a
  negative/conditional performance result.
