"""End-to-end smoke test for the vLLM v1 compressed CPU offloader.

The script runs several distinct prompts to create KV-cache pressure and then
replays the first prompt. Successful GPU->CPU and CPU->GPU jobs are written to
the configured JSONL metrics file by the custom connector.
"""

import argparse
import faulthandler
import json
import os
from pathlib import Path
import sys
import threading
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
existing_pythonpath = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    str(PROJECT_ROOT)
    if not existing_pythonpath
    else str(PROJECT_ROOT) + os.pathsep + existing_pythonpath
)

# FlashInfer 0.6.x does not currently identify Blackwell/SM 12 correctly in
# this environment.  Select vLLM's supported native sampler before importing
# vLLM; this does not change the attention or KV-cache transfer path.
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams


def partition_prompt_batches(prompts, *, batch_all=False, batch_sizes=None):
    """Partition prompts into explicit submission phases."""
    prompts = list(prompts)
    if batch_sizes:
        sizes = [int(size) for size in batch_sizes]
        if any(size <= 0 for size in sizes):
            raise ValueError("prompt batch sizes must be positive")
        if sum(sizes) != len(prompts):
            raise ValueError("prompt batch sizes must sum to the prompt count")
        batches = []
        offset = 0
        for size in sizes:
            batches.append(prompts[offset:offset + size])
            offset += size
        return batches
    if batch_all:
        return [prompts]
    return [[prompt] for prompt in prompts]


class H2DContender:
    """Continuously copy one pinned host buffer on a separate CUDA stream."""

    def __init__(self, size_mib: int, idle_us: float = 0.0, device_id: int = 0):
        self.size_bytes = int(size_mib) * 1024 * 1024
        self.idle_us = float(idle_us)
        self.device_id = int(device_id)
        self._ready = threading.Event()
        self._stop = threading.Event()
        self._thread = None
        self._error = None
        self._copies = 0
        self._elapsed_seconds = 0.0

    def _run(self):
        started = None
        try:
            torch.cuda.set_device(self.device_id)
            host = torch.empty(
                self.size_bytes, dtype=torch.uint8, pin_memory=True
            )
            device = torch.empty(
                self.size_bytes,
                dtype=torch.uint8,
                device=f"cuda:{self.device_id}",
            )
            stream = torch.cuda.Stream(device=self.device_id)
            self._ready.set()
            started = time.perf_counter()
            while not self._stop.is_set():
                with torch.cuda.stream(stream):
                    device.copy_(host, non_blocking=True)
                stream.synchronize()
                self._copies += 1
                if self.idle_us > 0:
                    self._stop.wait(self.idle_us / 1e6)
        except Exception as exc:
            self._error = exc
            self._ready.set()
        finally:
            if started is not None:
                self._elapsed_seconds = time.perf_counter() - started

    def start(self):
        if self.size_bytes <= 0:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="pcie-h2d-contender",
            daemon=True,
        )
        self._thread.start()
        if not self._ready.wait(timeout=60):
            raise RuntimeError("PCIe contender did not initialize")
        if self._error is not None:
            raise RuntimeError(
                "PCIe contender failed to initialize"
            ) from self._error

    def stop(self):
        if self._thread is None:
            return {
                "size_bytes": 0,
                "idle_us": self.idle_us,
                "bytes_copied": 0,
                "elapsed_seconds": 0.0,
                "throughput_gbps": 0.0,
            }
        self._stop.set()
        self._thread.join(timeout=60)
        if self._thread.is_alive():
            raise RuntimeError("PCIe contender did not stop")
        if self._error is not None:
            raise RuntimeError("PCIe contender failed") from self._error
        bytes_copied = self._copies * self.size_bytes
        return {
            "size_bytes": self.size_bytes,
            "idle_us": self.idle_us,
            "bytes_copied": bytes_copied,
            "elapsed_seconds": self._elapsed_seconds,
            "throughput_gbps": (
                bytes_copied * 8.0 / self._elapsed_seconds / 1e9
                if self._elapsed_seconds > 0 else 0.0
            ),
        }


def request_timing(request):
    metrics = request.metrics
    if metrics is None:
        return None
    generation_tokens = int(metrics.num_generation_tokens)
    decode_seconds = float(metrics.last_token_ts - metrics.first_token_ts)
    return {
        "ttft_ms": 1000 * float(metrics.first_token_latency),
        # vLLM 0.23 may expose arrival_time and token timestamps from different
        # clock domains. first_token_latency and the token timestamp delta are
        # individually valid, so compose E2E from those two durations.
        "e2e_ms": 1000 * (float(metrics.first_token_latency) + decode_seconds),
        "tpot_ms": (
            1000 * decode_seconds / max(generation_tokens - 1, 1)
        ),
        "generation_tokens": generation_tokens,
    }


def parse_bound_mapping(values, option_name):
    mapping = {}
    for item in values:
        try:
            bound_text, value_text = item.split("=", 1)
            bound = float(bound_text)
            value = float(value_text)
        except ValueError as exc:
            raise ValueError(
                f"{option_name} entries must use BOUND=VALUE: {item!r}"
            ) from exc
        if bound <= 0 or value <= 0:
            raise ValueError(f"{option_name} requires positive bounds and values")
        mapping[bound] = value
    return mapping


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--metrics", default="data/vllm_offload_smoke.jsonl")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=32 * 1024 * 1024)
    parser.add_argument("--cpu-offload-gb", type=float, default=1.0)
    parser.add_argument("--error-bound", type=float, default=1e-4)
    parser.add_argument(
        "--codec",
        choices=("raw", "cuszp", "int8", "zlib", "zstd", "lz4"),
        default="cuszp",
    )
    parser.add_argument(
        "--cuszp-mode",
        choices=("plain", "fixed", "outlier"),
        default="plain",
    )
    parser.add_argument(
        "--stock-offload",
        action="store_true",
        help="Use vLLM's stock CPU offloader for correctness diagnosis.",
    )
    parser.add_argument("--sensitivity-profile", default=None)
    parser.add_argument(
        "--sensitivity-policy",
        choices=("tolerant_only", "all_safe"),
        default="tolerant_only",
    )
    parser.add_argument("--adaptive-error-bound", action="store_true")
    parser.add_argument(
        "--async-store",
        action="store_true",
        help="Run GPU-to-CPU compression on a protected background CUDA stream.",
    )
    parser.add_argument(
        "--batch-restore-transfers",
        action="store_true",
        help="Queue pinned payload H2D copies together before decode.",
    )
    parser.add_argument(
        "--profile-restore-stages",
        action="store_true",
        help="Synchronize restore stages to profile H2D, decode, and scatter.",
    )
    parser.add_argument(
        "--pcie-contender-mib",
        type=int,
        default=0,
        help="Continuously copy this many MiB H2D during measured requests.",
    )
    parser.add_argument(
        "--pcie-contender-idle-us",
        type=float,
        default=0.0,
        help="Pause this many microseconds after each contender H2D copy.",
    )
    parser.add_argument(
        "--pcie-service-rate-gbps",
        type=float,
        default=0.0,
        help="Measured uncompressed PCIe service rate; required for adaptive mode.",
    )
    parser.add_argument(
        "--transfer-deadline-ms",
        type=float,
        default=0.0,
        help="Maximum desired KV transfer time; required for adaptive mode.",
    )
    parser.add_argument(
        "--adaptive-candidates",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4, 1e-3),
    )
    parser.add_argument(
        "--adaptive-cuszp-modes",
        nargs="+",
        choices=("plain", "fixed", "outlier"),
        default=None,
    )
    parser.add_argument("--cost-aware-restore", action="store_true")
    parser.add_argument("--restore-h2d-bandwidth-gbps", type=float, default=0.0)
    parser.add_argument(
        "--restore-ratio", nargs="*", default=[], metavar="BOUND=RATIO"
    )
    parser.add_argument(
        "--restore-decompression-gbps",
        nargs="*",
        default=[],
        metavar="BOUND=GBPS",
    )
    parser.add_argument("--restore-mode-profile", default=None)
    parser.add_argument("--restore-fixed-overhead-ms", type=float, default=0.0)
    parser.add_argument(
        "--restore-min-savings-fraction", type=float, default=0.05
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument(
        "--prompt-repeats",
        type=int,
        nargs="+",
        default=(32, 32, 32, 32, 32, 32),
        help="Per-request prefix sizes for stable or variable-pressure workloads.",
    )
    parser.add_argument(
        "--prompt-style",
        choices=("shared", "disjoint"),
        default="shared",
        help="Use legacy shared prefixes or distinct first-block prompt families.",
    )
    parser.add_argument(
        "--batch-prompts",
        action="store_true",
        help="Submit the configured prompts concurrently instead of sequentially.",
    )
    parser.add_argument(
        "--prompt-batch-sizes",
        type=int,
        nargs="+",
        default=None,
        help="Explicit prompt submission phases, for example 1 1 6.",
    )
    parser.add_argument(
        "--replay-all",
        action="store_true",
        help="Replay every pressure prompt and report token-level agreement.",
    )
    parser.add_argument("--summary", default=None)
    parser.add_argument("--append-metrics", action="store_true")
    parser.add_argument(
        "--warmup-offload",
        action="store_true",
        help="Run an unmeasured pressure workload before the recorded requests.",
    )
    parser.add_argument(
        "--startup-dump-seconds",
        type=int,
        default=0,
        help="Dump Python thread stacks after this many seconds (0 disables it).",
    )
    args = parser.parse_args()
    if args.pcie_contender_mib < 0:
        parser.error("--pcie-contender-mib cannot be negative")
    if args.pcie_contender_idle_us < 0:
        parser.error("--pcie-contender-idle-us cannot be negative")
    if args.prompt_batch_sizes:
        if any(size <= 0 for size in args.prompt_batch_sizes):
            parser.error("--prompt-batch-sizes values must be positive")
        if sum(args.prompt_batch_sizes) != len(args.prompt_repeats):
            parser.error("--prompt-batch-sizes must sum to prompt count")
    try:
        restore_ratios = parse_bound_mapping(
            args.restore_ratio, "--restore-ratio"
        )
        restore_decompression_gbps = parse_bound_mapping(
            args.restore_decompression_gbps, "--restore-decompression-gbps"
        )
    except ValueError as exc:
        parser.error(str(exc))
    restore_mode_profiles = {}
    if args.restore_mode_profile:
        try:
            restore_mode_profiles = json.loads(
                Path(args.restore_mode_profile).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --restore-mode-profile: {exc}")
        if not isinstance(restore_mode_profiles, dict):
            parser.error("--restore-mode-profile must contain a JSON object")
    if args.cost_aware_restore:
        if not args.adaptive_error_bound:
            parser.error("--cost-aware-restore requires --adaptive-error-bound")
        if args.restore_h2d_bandwidth_gbps <= 0:
            parser.error("--cost-aware-restore requires positive H2D bandwidth")
        if not restore_mode_profiles and (
            not restore_ratios or not restore_decompression_gbps):
            parser.error(
                "cost-aware restore requires a mode profile or bound mappings"
            )

    if args.adaptive_error_bound:
        if not args.sensitivity_profile:
            parser.error("--adaptive-error-bound requires --sensitivity-profile")
        try:
            sensitivity_data = json.loads(
                Path(args.sensitivity_profile).read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as exc:
            parser.error(f"cannot read --sensitivity-profile: {exc}")
        profile_model = sensitivity_data.get("_metadata", {}).get("model")
        if profile_model and profile_model != args.model:
            parser.error(
                f"sensitivity profile is for {profile_model}, not {args.model}"
            )
        if args.pcie_service_rate_gbps <= 0 or args.transfer_deadline_ms <= 0:
            parser.error(
                "adaptive mode requires positive --pcie-service-rate-gbps "
                "and --transfer-deadline-ms"
            )

    if args.startup_dump_seconds > 0:
        faulthandler.dump_traceback_later(args.startup_dump_seconds, repeat=True)

    metrics_path = str(Path(args.metrics).resolve())
    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    if not args.append_metrics and Path(metrics_path).exists():
        Path(metrics_path).unlink()

    connector_extra_config = {
        "cpu_bytes_to_use": int(args.cpu_offload_gb * (1 << 30)),
        "store_threshold": 0,
    }
    if not args.stock_offload:
        connector_extra_config.update(
            {
                "spec_name": "CompressedCPUOffloadingSpec",
                "spec_module_path": (
                    "integration.compression_pipeline.vllm_v1_compressed_offload"
                ),
                "error_bound": args.error_bound,
                "codec": args.codec,
                "cuszp_mode": args.cuszp_mode,
                "metrics_path": metrics_path,
                "async_store": args.async_store,
                "profile_restore_stages": args.profile_restore_stages,
                "batch_restore_transfers": args.batch_restore_transfers,
            }
        )
    if args.sensitivity_profile:
        connector_extra_config["sensitivity_profile"] = str(
            Path(args.sensitivity_profile).resolve()
        )
        connector_extra_config["sensitivity_policy"] = args.sensitivity_policy
    if args.adaptive_error_bound:
        connector_extra_config.update(
            {
                "adaptive_error_bound": True,
                "pcie_service_rate_gbps": args.pcie_service_rate_gbps,
                "transfer_deadline_ms": args.transfer_deadline_ms,
                "adaptive_candidates": list(args.adaptive_candidates),
                "adaptive_cuszp_modes": (
                    args.adaptive_cuszp_modes or [args.cuszp_mode]
                ),
            }
        )

    if args.cost_aware_restore:
        connector_extra_config.update(
            {
                "cost_aware_restore": True,
                "restore_h2d_bandwidth_gbps": args.restore_h2d_bandwidth_gbps,
                "restore_compression_ratios": restore_ratios,
                "restore_decompression_gbps": restore_decompression_gbps,
                "restore_fixed_overhead_ms": args.restore_fixed_overhead_ms,
                "restore_min_savings_fraction": args.restore_min_savings_fraction,
                "restore_mode_profiles": restore_mode_profiles,
            }
        )

    kv_transfer_config = {
        "kv_connector": "OffloadingConnector",
        "kv_role": "kv_both",
        "kv_connector_extra_config": connector_extra_config,
    }
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
        kv_offloading_size=args.cpu_offload_gb,
        kv_transfer_config=kv_transfer_config,
        enforce_eager=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        disable_log_stats=False,
        seed=0,
    )
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)
    base = "KV cache offloading over PCIe is useful because "
    disjoint_seeds = [
        "Network packet scheduling reduces congestion in datacenter switches. ",
        "Astronomers measure distant galaxies using calibrated spectral sensors. ",
        "Marine ecosystems depend on stable temperature and nutrient cycles. ",
        "Compiler optimization transforms programs while preserving exact semantics. ",
        "Urban rail planning balances passenger demand and station capacity. ",
        "Medical imaging reconstructs internal structures from noisy measurements. ",
    ]
    if args.warmup_offload:
        warmup_repeat = max(args.prompt_repeats)
        warmup_prompts = [
            (base + f"warmup stream {idx}. ") * warmup_repeat
            for idx in range(6)
        ]
        for prompt in warmup_prompts:
            llm.generate([prompt], sampling)
        # MetricsRecorder opens the file per event, so removing the completed
        # warm-up trace safely starts a clean measured interval.
        if Path(metrics_path).exists():
            Path(metrics_path).unlink()
    prompts = [
        (
            (base + f"experiment stream {idx}. ")
            if args.prompt_style == "shared"
            else disjoint_seeds[idx % len(disjoint_seeds)]
        )
        * repeat
        for idx, repeat in enumerate(args.prompt_repeats)
    ]
    contender = H2DContender(
        args.pcie_contender_mib,
        idle_us=args.pcie_contender_idle_us,
    )
    contender.start()
    initial_requests = []
    for prompt_batch in partition_prompt_batches(
        prompts,
        batch_all=args.batch_prompts,
        batch_sizes=args.prompt_batch_sizes,
    ):
        initial_requests.extend(llm.generate(prompt_batch, sampling))
    initial_outputs = [request.outputs[0] for request in initial_requests]
    replay_prompts = prompts if args.replay_all else prompts[:1]
    replay_requests = []
    replay_batch_sizes = (
        args.prompt_batch_sizes if len(replay_prompts) == len(prompts) else None
    )
    for prompt_batch in partition_prompt_batches(
        replay_prompts,
        batch_all=args.batch_prompts,
        batch_sizes=replay_batch_sizes,
    ):
        replay_requests.extend(llm.generate(prompt_batch, sampling))
    replay_outputs = [request.outputs[0] for request in replay_requests]
    contender_stats = contender.stop()
    initial_timings = [
        timing for request in initial_requests
        if (timing := request_timing(request)) is not None
    ]
    replay_timings = [
        timing for request in replay_requests
        if (timing := request_timing(request)) is not None
    ]
    if len(initial_timings) != len(initial_requests) or len(replay_timings) != len(
        replay_requests
    ):
        raise RuntimeError(
            "vLLM did not return request timing metrics for every request"
        )

    agreement = []
    for initial, replayed in zip(initial_outputs, replay_outputs):
        reference_ids = list(initial.token_ids)
        replay_ids = list(replayed.token_ids)
        denominator = max(len(reference_ids), len(replay_ids), 1)
        matching = sum(
            left == right for left, right in zip(reference_ids, replay_ids)
        )
        agreement.append(
            {
                "reference_token_ids": reference_ids,
                "replay_token_ids": replay_ids,
                "token_match_rate": matching / denominator,
                "exact_match": reference_ids == replay_ids,
                "reference_text": initial.text,
                "replay_text": replayed.text,
            }
        )

    events = []
    if os.path.exists(metrics_path):
        with open(metrics_path, "r", encoding="utf-8") as fh:
            events = [json.loads(line) for line in fh if line.strip()]
    successful = [event for event in events if event.get("success")]
    directions = {event.get("direction") for event in successful}
    summary = {
        "metrics_path": metrics_path,
        "num_events": len(events),
        "num_successful": len(successful),
        "directions": sorted(direction for direction in directions if direction),
        "codec": args.codec,
        "cuszp_mode": args.cuszp_mode,
        "adaptive_cuszp_modes": (
            args.adaptive_cuszp_modes or [args.cuszp_mode]
        ),
        "restore_mode_profile": args.restore_mode_profile,
        "stock_offload": args.stock_offload,
        "error_bound": args.error_bound,
        "adaptive_error_bound": args.adaptive_error_bound,
        "cost_aware_restore": args.cost_aware_restore,
        "async_store": args.async_store,
        "profile_restore_stages": args.profile_restore_stages,
        "batch_restore_transfers": args.batch_restore_transfers,
        "pcie_contender_mib": args.pcie_contender_mib,
        "pcie_contender_idle_us": args.pcie_contender_idle_us,
        "pcie_contender": contender_stats,
        "prompt_repeats": list(args.prompt_repeats),
        "prompt_style": args.prompt_style,
        "batch_prompts": args.batch_prompts,
        "prompt_batch_sizes": args.prompt_batch_sizes,
        "warmup_offload": args.warmup_offload,
        "num_replays": len(agreement),
        "mean_token_match_rate": (
            sum(item["token_match_rate"] for item in agreement) / len(agreement)
        ),
        "exact_match_rate": (
            sum(item["exact_match"] for item in agreement) / len(agreement)
        ),
        "initial_request_timings": initial_timings,
        "replay_request_timings": replay_timings,
        "mean_initial_e2e_ms": (
            sum(item["e2e_ms"] for item in initial_timings) / len(initial_timings)
        ),
        "mean_replay_e2e_ms": (
            sum(item["e2e_ms"] for item in replay_timings) / len(replay_timings)
        ),
        "mean_replay_ttft_ms": (
            sum(item["ttft_ms"] for item in replay_timings) / len(replay_timings)
        ),
        "mean_replay_tpot_ms": (
            sum(item["tpot_ms"] for item in replay_timings) / len(replay_timings)
        ),
        "agreement": agreement,
    }
    print(json.dumps(summary, indent=2))
    if args.summary:
        summary_path = Path(args.summary).resolve()
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not args.stock_offload and "gpu_to_cpu" not in directions:
        raise RuntimeError("No successful GPU-to-CPU compressed offload was observed")


if __name__ == "__main__":
    main()
