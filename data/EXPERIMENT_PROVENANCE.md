# Formal INFOCOM experiment provenance

Only paths admitted by "reproducibility/formal_artifacts.json" are formal paper
evidence. "reproducibility/formal_data.sha256" authenticates every admitted
JSON, JSONL, Markdown, prompt, and calibration profile.

## Runtime experiments

All reported latency, throughput, compression, and quality values come from
real vLLM 0.23 execution on the reviewed RTX 5080 / Core Ultra 7 265KF system.
The canonical suites are:

- "gate_d_fair_async_probe": five interleaved trials for eight models and five
  matched transfer methods.
- "qwen1.5b_real_concurrency_v3": real simultaneous bursts at 2, 4, 8, and 16.
- "qwen1.5b_open_loop_arrival_v1": fixed arrivals at 2, 4, and 6 request/s.
- "qwen0.5b_open_loop_arrival_v1": fixed arrivals at 8 request/s.
- "infocom_long_context_quality_v1/five_trial": five-trial retrieval-quality
  comparison.
- "vllm_qwen1.5b_4k_gate_c_packed_mixed21_complete_2trial": two-trial adaptive
  mechanism pilot; "cost_aware_restore" is false.

The main concurrency and open-loop suites do not use a synthetic PCIe
contender. Logs, binaries, caches, exploratory probes, and historical invalid
runs are excluded by ".gitignore" and are not paper evidence.

## Adaptive calibration

"infocom_calibration_prompts.json" defines eight domains. The eight files under
"qwen2.5-1.5b_multi_prompt_sensitivity_1e-5_1e-4_per_prompt/" are the raw
teacher-forced layer sweeps at 1e-5 and 1e-4. The merge uses maximum KL,
minimum top-1 agreement, KL <= 0.01, and top-1 >= 0.875. The deterministic
selection retains 21 layers at 1e-4 and seven at 1e-5.

Rebuild the merge and final selection with "./run.sh calibrate"; rerun the GPU
measurements with "./run.sh calibrate-fresh".

## Acceptance procedure

Canonical experiments use "--resume" only to finish missing trials. Independent
reproduction uses "REPRO_OUT_ROOT=... ./run.sh rerun", which rejects nonempty
directories and never reuses canonical trial files. After deliberately
replacing canonical measurements, review all changes and rebuild the hash
manifest explicitly.
