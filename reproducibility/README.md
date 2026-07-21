# Reproducibility snapshot

This directory freezes the software, third-party codec, model revisions,
hardware, commands, and hashes used by the INFOCOM manuscript source
`paper/conference_101719.tex`.

## What is pinned

- `environment.json`: Python packages, build tools, cuSZp commit, and reviewed
  RTX 5080 / Core Ultra 7 265KF platform.
- `models.json`: exact Hugging Face snapshot revisions for all eight Gate D
  models.
- `formal_artifacts.json`: the measured data directories admitted as paper
  evidence. Other directories under `data/` are historical or exploratory.
- `formal_data.sha256`: hashes for every admitted JSON, JSONL, and provenance
  document.
- `provenance.py`: runtime provenance embedded by new repeated experiments.
- `verify_environment.py`: fail-closed environment and result verification.

## Setup and verification

Create the Python environment and install the exact versions:

```bash
python3.12 -m venv venv
venv/bin/pip install -r requirements.txt
```

The reviewed machine uses cuSZp commit
`f581dcf329c907c320f4743a9c6e7ee2fb9c5494` at `/opt/cuSZp`. For a
user-writable checkout:

```bash
CUSZP_ROOT="$PWD/.deps/cuSZp" reproducibility/install_cuszp.sh
CUSZP_ROOT="$PWD/.deps/cuSZp" SKIP_BUILD=0 ./run.sh validate
```

Normal validation does not launch model inference:

```bash
./run.sh validate
./run.sh dry-run
```

`./run.sh calibrate` rebuilds the adaptive mixed profile from the admitted
per-prompt measurements; `calibrate-fresh` reruns all eight GPU sweeps.
`./run.sh experiments` resumes canonical trials and can take many hours.
For an independent no-reuse run, choose a new directory:

```bash
REPRO_OUT_ROOT=data/reproductions/run-01 ./run.sh rerun
```

`./run.sh assets` regenerates figures and synchronizes the six marked inline
result blocks in the standalone paper. `./run.sh verify-paper` checks exact
paper/data equality. No target compiles a PDF. After changing canonical
measurements, review them before running
`reproducibility/build_hash_manifest.py --write`.

## Protocol version note

The published Gate D snapshot used the historical six-seed/eight-slot prompt
semantics. It is explicitly named `legacy_v1` and maps to
`legacy_disjoint_v1`. The corrected eight-unique-stream workload is
`corrected_v2` and is not silently substituted for the published data.
The same legacy prompt semantics reproduce the adaptive pilot whose old JSON
recorded the then-current label `disjoint`.
