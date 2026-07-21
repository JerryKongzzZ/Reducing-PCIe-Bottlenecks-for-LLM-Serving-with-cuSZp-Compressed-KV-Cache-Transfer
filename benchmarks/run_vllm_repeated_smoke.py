"""Run isolated, warmed vLLM offload trials and aggregate 95% confidence intervals."""

import argparse
import json
import math
import os
import shlex
from pathlib import Path
import statistics
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from reproducibility.provenance import collect_runtime_provenance

SMOKE = REPO_ROOT / "benchmarks" / "smoke_vllm_compressed_offload.py"


def ci95(values):
    if len(values) < 2:
        return 0.0
    # Two-sided Student-t critical values for the small trial counts used here.
    critical = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}.get(
        len(values), 1.96
    )
    return critical * statistics.stdev(values) / math.sqrt(len(values))


def describe(values):
    return {
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "ci95_half_width": ci95(values),
        "trials": list(values),
    }


def percentile(values, quantile):
    """Return an R-7 linearly interpolated percentile."""
    if not values:
        raise ValueError("percentile requires at least one value")
    if quantile < 0.0 or quantile > 1.0:
        raise ValueError("quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_request_timings(
    timings,
    *,
    slo_ttft_ms=0.0,
    slo_e2e_ms=0.0,
):
    """Pool request timings before computing tails and SLO attainment."""
    timings = list(timings)
    result = {"count": len(timings)}
    if not timings:
        return result
    for field in ("ttft_ms", "e2e_ms", "tpot_ms"):
        values = [float(item[field]) for item in timings]
        result[field] = {
            "mean": statistics.mean(values),
            "p50": percentile(values, 0.50),
            "p95": percentile(values, 0.95),
            "p99": percentile(values, 0.99),
            "max": max(values),
        }
    result["slo"] = {}
    for field, threshold in (
        ("ttft_ms", float(slo_ttft_ms)),
        ("e2e_ms", float(slo_e2e_ms)),
    ):
        if threshold > 0.0:
            within = sum(float(item[field]) <= threshold for item in timings)
            result["slo"][field] = {
                "threshold_ms": threshold,
                "attainment": within / len(timings),
                "violations": len(timings) - within,
            }
    return result


def method_error_bound(method, default_error_bound):
    """Return a fixed bound only for methods that actually use one."""
    if method == "async_cuszp_1e-5":
        return 1e-5
    if method == "async_cuszp_1e-3":
        return 1e-3
    if method in ("cuszp", "async_cuszp"):
        return default_error_bound
    return None


def trial_metrics(metrics_path, summary_path, stock_offload=False):
    events = [] if stock_offload else [
        json.loads(line)
        for line in metrics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    saves = [
        event for event in events
        if event.get("success") and event.get("direction") == "gpu_to_cpu"
    ]
    loads = [
        event for event in events
        if event.get("success") and event.get("direction") == "cpu_to_gpu"
    ]
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not stock_offload and (not saves or not loads):
        raise RuntimeError(f"missing transfer direction in {metrics_path}")
    result = {
        "compression_ratio": (
            1.0 if stock_offload else statistics.mean(
                event["compression_ratio"] for event in saves
            )
        ),
        "configured_error_bound": summary["error_bound"],
        "token_match_rate": summary["mean_token_match_rate"],
        "exact_match_rate": summary["exact_match_rate"],
        "initial_e2e_ms": summary["mean_initial_e2e_ms"],
        "replay_e2e_ms": summary["mean_replay_e2e_ms"],
        "replay_ttft_ms": summary["mean_replay_ttft_ms"],
        "replay_tpot_ms": summary["mean_replay_tpot_ms"],
        "pcie_contender_gbps": summary.get(
            "pcie_contender", {}
        ).get("throughput_gbps", 0.0),
        "gpu_to_cpu_events": len(saves),
        "cpu_to_gpu_events": len(loads),
        "adaptive_states": [
            event["adaptive"]["state"]
            for event in saves
            if event.get("adaptive")
        ],
        "adaptive_state_changes": sum(
            bool(event["adaptive"].get("state_changed"))
            for event in saves
            if event.get("adaptive")
        ),
        "mean_pressure": (
            statistics.mean(
                event["adaptive"]["pressure"]
                for event in saves
                if event.get("adaptive")
            )
            if any(event.get("adaptive") for event in saves)
            else None
        ),
        "_initial_request_timings": list(
            summary.get("initial_request_timings", [])
        ),
        "_replay_request_timings": list(
            summary.get("replay_request_timings", [])
        ),
    }
    optional_summary_metrics = {
        "initial_wall_ms": "initial_wall_ms",
        "replay_wall_ms": "replay_wall_ms",
        "initial_requests_per_second": "initial_requests_per_second",
        "replay_requests_per_second": "replay_requests_per_second",
        "initial_output_tokens_per_second": "initial_output_tokens_per_second",
        "replay_output_tokens_per_second": "replay_output_tokens_per_second",
        "initial_task_accuracy": "initial_task_accuracy",
        "replay_task_accuracy": "replay_task_accuracy",
    }
    for output_key, summary_key in optional_summary_metrics.items():
        if summary_key in summary and summary[summary_key] is not None:
            result[output_key] = summary[summary_key]

    if not stock_offload:
        result.update(
            {
                "gpu_to_cpu_ms": 1000 * statistics.mean(
                    event["elapsed_seconds"] for event in saves
                ),
                "cpu_to_gpu_ms": 1000 * statistics.mean(
                    event["elapsed_seconds"] for event in loads
                ),
            }
        )
    profiled_loads = [event for event in loads if event.get("restore_stages")]
    if profiled_loads:
        stage_fields = {
            "restore_cpu_decode_ms": "cpu_decode_seconds",
            "restore_h2d_ms": "h2d_seconds",
            "restore_gpu_decode_ms": "gpu_decode_seconds",
            "restore_scatter_ms": "scatter_seconds",
        }
        for output_key, event_key in stage_fields.items():
            result[output_key] = 1000 * statistics.mean(
                event["restore_stages"][event_key] for event in profiled_loads
            )
        result["restore_profiled_total_ms"] = sum(
            result[key] for key in stage_fields
        )
        total_h2d_seconds = sum(
            event["restore_stages"]["h2d_seconds"] for event in profiled_loads
        )
        total_restore_seconds = sum(
            event["elapsed_seconds"] for event in profiled_loads
        )
        total_transferred_bytes = sum(
            event["transferred_bytes"] for event in profiled_loads
        )
        result["restore_h2d_fraction"] = (
            total_h2d_seconds / max(total_restore_seconds, 1e-12)
        )
        result["effective_h2d_gbps"] = (
            total_transferred_bytes * 8.0 / total_h2d_seconds / 1e9
            if total_h2d_seconds > 0 else 0.0
        )
        result["mean_restore_original_bytes"] = statistics.mean(
            event["original_bytes"] for event in profiled_loads
        )
        result["mean_restore_transferred_bytes"] = statistics.mean(
            event["transferred_bytes"] for event in profiled_loads
        )
        decoded_loads = [
            event
            for event in profiled_loads
            if event["restore_stages"]["gpu_decode_seconds"] > 0
        ]
        result["restore_decompression_gbps"] = (
            sum(event["original_bytes"] for event in decoded_loads) * 8.0
            / sum(
                event["restore_stages"]["gpu_decode_seconds"]
                for event in decoded_loads
            ) / 1e9
            if decoded_loads else 0.0
        )

    return result


def interleaved_trial_schedule(methods, num_trials):
    """Pair methods within each trial to reduce time/temperature drift."""
    return [
        (method, trial)
        for trial in range(1, num_trials + 1)
        for method in methods
    ]
def paired_descriptions(baseline_trials, candidate_trials, metric_keys):
    """Describe candidate-minus-baseline differences for paired trials."""
    if len(baseline_trials) != len(candidate_trials):
        raise ValueError("paired trial counts differ")
    return {
        key: describe([
            candidate[key] - baseline[key]
            for baseline, candidate in zip(
                baseline_trials, candidate_trials
            )
        ])
        for key in metric_keys
        if all(key in trial for trial in baseline_trials + candidate_trials)
    }



def evaluate_quality_gate(
    all_results,
    *,
    baseline_method,
    min_token_match_rate=0.0,
    min_exact_match_rate=0.0,
    max_token_match_drop=0.0,
    max_exact_match_drop=0.0,
    min_task_accuracy=0.0,
    max_task_accuracy_drop=0.0,
):
    """Require every trial to meet absolute and baseline-relative quality."""
    if baseline_method not in all_results:
        raise ValueError(f"quality baseline method is missing: {baseline_method}")
    baseline_trials = all_results[baseline_method]["trial_details"]
    failures = []
    methods = {}
    for method, result in all_results.items():
        trials = result["trial_details"]
        if len(trials) != len(baseline_trials):
            raise ValueError(f"quality trial count differs for {method}")
        trial_checks = []
        for index, (baseline, candidate) in enumerate(
            zip(baseline_trials, trials), start=1
        ):
            token_floor = max(
                min_token_match_rate,
                baseline["token_match_rate"] - max_token_match_drop,
            )
            exact_floor = max(
                min_exact_match_rate,
                baseline["exact_match_rate"] - max_exact_match_drop,
            )
            task_available = (
                "replay_task_accuracy" in baseline
                and "replay_task_accuracy" in candidate
            )
            task_floor = (
                max(
                    min_task_accuracy,
                    baseline["replay_task_accuracy"] - max_task_accuracy_drop,
                )
                if task_available else None
            )
            task_passed = (
                candidate["replay_task_accuracy"] >= task_floor
                if task_available else True
            )
            passed = (
                candidate["token_match_rate"] >= token_floor
                and candidate["exact_match_rate"] >= exact_floor
                and task_passed
            )
            check = {
                "trial": index,
                "token_match_rate": candidate["token_match_rate"],
                "token_match_floor": token_floor,
                "exact_match_rate": candidate["exact_match_rate"],
                "exact_match_floor": exact_floor,
                "passed": passed,
            }
            if task_available:
                check.update(
                    {
                        "replay_task_accuracy": candidate["replay_task_accuracy"],
                        "replay_task_accuracy_floor": task_floor,
                    }
                )
            trial_checks.append(check)
            if not passed:
                failures.append({"method": method, **check})
        methods[method] = {
            "passed": all(item["passed"] for item in trial_checks),
            "trials": trial_checks,
        }
    return {
        "passed": not failures,
        "baseline_method": baseline_method,
        "min_token_match_rate": min_token_match_rate,
        "min_exact_match_rate": min_exact_match_rate,
        "max_token_match_drop": max_token_match_drop,
        "max_exact_match_drop": max_exact_match_drop,
        "min_task_accuracy": min_task_accuracy,
        "max_task_accuracy_drop": max_task_accuracy_drop,
        "methods": methods,
        "failures": failures,
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=(
            "stock", "raw", "async_raw", "cuszp", "async_cuszp", "async_cuszp_1e-5",
            "async_cuszp_1e-3", "int8", "async_int8", "zlib",
            "async_zlib", "zstd", "async_zstd", "lz4", "async_lz4",
            "adaptive", "async_adaptive"
        ),
        default=("raw", "cuszp", "int8", "zlib", "adaptive"),
    )
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--trial-order",
        choices=("interleaved", "grouped"),
        default="interleaved",
        help="Interleave methods by trial to reduce thermal and temporal drift.",
    )
    parser.add_argument(
        "--trial-offset", type=int, default=0, help=argparse.SUPPRESS
    )
    parser.add_argument(
        "--internal-single",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--aggregate-only",
        action="store_true",
        help="Recompute aggregate.json from existing trial metrics and summaries.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the top-level command without launching trials.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete trial metric/summary pairs and run only missing trials.",
    )
    parser.add_argument("--out-dir", default="data/vllm_repeated_smoke")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=32 * 1024 * 1024
    )
    parser.add_argument(
        "--prompt-repeats",
        type=int,
        nargs="+",
        default=(32, 32, 32, 32, 32, 32),
    )
    parser.add_argument(
        "--prompt-style",
        choices=("shared", "legacy_disjoint_v1", "disjoint"),
        default="shared",
    )
    parser.add_argument("--prompt-file", default=None)
    parser.add_argument("--batch-prompts", action="store_true")
    parser.add_argument("--interarrival-ms", type=float, default=0.0)
    parser.add_argument("--prompt-batch-sizes", type=int, nargs="+", default=None)
    parser.add_argument("--profile-restore-stages", action="store_true")
    parser.add_argument("--batch-restore-transfers", action="store_true")
    parser.add_argument("--pcie-contender-mib", type=int, default=0)
    parser.add_argument("--pcie-contender-idle-us", type=float, default=0.0)
    parser.add_argument("--error-bound", type=float, default=1e-4)
    parser.add_argument(
        "--cuszp-mode",
        choices=("plain", "fixed", "outlier"),
        default="plain",
    )
    parser.add_argument(
        "--adaptive-cuszp-modes",
        nargs="+",
        choices=("plain", "fixed", "outlier"),
        default=("plain", "fixed", "outlier"),
    )
    parser.add_argument("--adaptive-profile")
    parser.add_argument(
        "--adaptive-candidates",
        type=float,
        nargs="+",
        default=(1e-5, 1e-4),
        help="Allowed bounds; 1e-3 requires an explicit override.",
    )
    parser.add_argument("--pcie-service-rate-gbps", type=float, default=1.5676)
    parser.add_argument("--transfer-deadline-ms", type=float, default=20.0)
    parser.add_argument("--cpu-offload-gb", type=float, default=1.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.75)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--slo-ttft-ms", type=float, default=0.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=0.0)
    parser.add_argument("--trial-timeout-seconds", type=int, default=900)
    parser.add_argument("--cost-aware-restore", action="store_true")
    parser.add_argument("--restore-h2d-bandwidth-gbps", type=float, default=0.0)
    parser.add_argument("--quality-gate", action="store_true")
    parser.add_argument("--quality-baseline-method")
    parser.add_argument("--quality-min-token-match-rate", type=float, default=0.0)
    parser.add_argument("--quality-min-exact-match-rate", type=float, default=0.0)
    parser.add_argument("--quality-max-token-match-drop", type=float, default=0.0)
    parser.add_argument("--quality-max-exact-match-drop", type=float, default=0.0)
    parser.add_argument("--quality-min-task-accuracy", type=float, default=0.0)
    parser.add_argument("--quality-max-task-accuracy-drop", type=float, default=0.0)
    parser.add_argument("--restore-ratio", nargs="*", default=[])
    parser.add_argument("--restore-decompression-gbps", nargs="*", default=[])
    parser.add_argument("--restore-fixed-overhead-ms", type=float, default=0.0)
    parser.add_argument("--restore-mode-profile", default=None)
    parser.add_argument(
        "--restore-min-savings-fraction", type=float, default=0.05
    )
    args = parser.parse_args()
    if args.trials < 2 and not args.internal_single:
        parser.error("at least two trials are required for confidence intervals")
    if args.trial_offset < 0:
        parser.error("--trial-offset cannot be negative")
    if args.pcie_contender_mib < 0:
        parser.error("--pcie-contender-mib cannot be negative")
    if args.pcie_contender_idle_us < 0:
        parser.error("--pcie-contender-idle-us cannot be negative")
    if args.slo_ttft_ms < 0 or args.slo_e2e_ms < 0:
        parser.error("SLO thresholds cannot be negative")
    if args.interarrival_ms < 0:
        parser.error("--interarrival-ms cannot be negative")
    if any("adaptive" in method for method in args.methods) and not args.adaptive_profile:
        parser.error("--adaptive-profile is required for the adaptive method")
    quality_values = (
        args.quality_min_token_match_rate,
        args.quality_min_exact_match_rate,
        args.quality_max_token_match_drop,
        args.quality_max_exact_match_drop,
        args.quality_min_task_accuracy,
        args.quality_max_task_accuracy_drop,
    )
    if any(value < 0.0 or value > 1.0 for value in quality_values):
        parser.error("quality thresholds and allowed drops must be in [0, 1]")
    if args.quality_gate and not args.internal_single:
        quality_baseline = args.quality_baseline_method or args.methods[0]
        if quality_baseline not in args.methods:
            parser.error("--quality-baseline-method must be included in --methods")
    if args.cost_aware_restore:
        if not any("adaptive" in method for method in args.methods):
            parser.error("--cost-aware-restore requires an adaptive method")
        if args.restore_h2d_bandwidth_gbps <= 0:
            parser.error("cost-aware restore requires positive H2D bandwidth")
        if not args.restore_mode_profile and (
            not args.restore_ratio or not args.restore_decompression_gbps):
            parser.error("cost-aware restore requires calibration mappings")
    if args.dry_run:
        if args.aggregate_only or args.internal_single:
            parser.error("--dry-run is only valid for a top-level experiment")
        print(
            "validated formal command: "
            + shlex.join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
            flush=True,
        )
        return 0
    if (
        args.trial_order == "interleaved"
        and not args.aggregate_only
        and not args.internal_single
    ):
        forwarded = sys.argv[1:]
        for method, trial in interleaved_trial_schedule(
            args.methods, args.trials
        ):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                *forwarded,
                "--methods", method,
                "--trials", "1",
                "--trial-offset", str(trial - 1),
                "--trial-order", "grouped",
                "--internal-single",
            ]
            print(
                f"dispatching interleaved {method} trial {trial}/{args.trials}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        args.aggregate_only = True

    out_dir = (REPO_ROOT / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = dict(os.environ)
    environment.update(
        {"PYTHONUNBUFFERED": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    )
    all_results = {}
    for method in args.methods:
        trials = []
        pooled_initial_timings = []
        pooled_replay_timings = []
        for trial in range(1, args.trials + 1):
            trial_number = trial + args.trial_offset
            stem = f"{method}_trial_{trial_number}"
            metrics_path = out_dir / f"{stem}.jsonl"
            summary_path = out_dir / f"{stem}_summary.json"
            log_path = out_dir / f"{stem}.log"
            if method in (
                "cuszp", "async_cuszp", "async_cuszp_1e-5",
                "async_cuszp_1e-3", "adaptive", "async_adaptive"
            ):
                codec = "cuszp"
            elif method == "async_raw":
                codec = "raw"
            elif method in (
                "async_int8", "async_zlib", "async_zstd", "async_lz4"
            ):
                codec = method.removeprefix("async_")
            else:
                codec = "raw" if method == "stock" else method
            command = [
                sys.executable,
                str(SMOKE),
                "--model", args.model,
                "--codec", codec,
                "--cuszp-mode", args.cuszp_mode,
                "--error-bound", str(
                    method_error_bound(method, args.error_bound) or args.error_bound
                ),
                "--max-model-len", str(args.max_model_len),
                "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
                "--cpu-offload-gb", str(args.cpu_offload_gb),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-tokens", str(args.max_tokens),
                "--prompt-repeats", *[str(value) for value in args.prompt_repeats],
                "--prompt-style", args.prompt_style,
                "--metrics", str(metrics_path),
                "--summary", str(summary_path),
                "--replay-all",
                "--warmup-offload",
            ]
            if args.interarrival_ms > 0:
                command.extend(["--interarrival-ms", str(args.interarrival_ms)])
            if args.prompt_file:
                command.extend(["--prompt-file", str(Path(args.prompt_file).resolve())])
            if method == "stock":
                command.append("--stock-offload")
            if args.batch_prompts:
                command.append("--batch-prompts")
            if args.prompt_batch_sizes:
                command.extend([
                    "--prompt-batch-sizes",
                    *[str(value) for value in args.prompt_batch_sizes],
                ])
            if args.profile_restore_stages and method != "stock":
                command.append("--profile-restore-stages")
            if args.batch_restore_transfers and method != "stock":
                command.append("--batch-restore-transfers")
            if args.pcie_contender_mib:
                command.extend(
                    ["--pcie-contender-mib", str(args.pcie_contender_mib)]
                )
                if args.pcie_contender_idle_us:
                    command.extend([
                        "--pcie-contender-idle-us",
                        str(args.pcie_contender_idle_us),
                    ])
            if method in (
                "async_raw", "async_cuszp", "async_cuszp_1e-5", "async_cuszp_1e-3",
                "async_int8", "async_zlib", "async_zstd", "async_lz4",
                "async_adaptive"
            ):
                command.append("--async-store")
            if method in ("adaptive", "async_adaptive"):
                command.extend(
                    [
                        "--sensitivity-profile", str(Path(args.adaptive_profile).resolve()),
                        "--sensitivity-policy", "all_safe",
                        "--adaptive-error-bound",
                        "--pcie-service-rate-gbps", str(args.pcie_service_rate_gbps),
                        "--transfer-deadline-ms", str(args.transfer_deadline_ms),
                        "--adaptive-candidates",
                        *[str(value) for value in args.adaptive_candidates],
                        "--adaptive-cuszp-modes",
                        *args.adaptive_cuszp_modes,
                    ]
                )
            if args.cost_aware_restore:
                command.extend(
                    [
                        "--cost-aware-restore",
                        "--restore-h2d-bandwidth-gbps",
                        str(args.restore_h2d_bandwidth_gbps),
                        "--restore-ratio",
                        *args.restore_ratio,
                        "--restore-decompression-gbps",
                        *args.restore_decompression_gbps,
                        "--restore-fixed-overhead-ms",
                        str(args.restore_fixed_overhead_ms),
                        "--restore-min-savings-fraction",
                        str(args.restore_min_savings_fraction),
                    ]
                )
                if args.restore_mode_profile:
                    command.extend(
                        ["--restore-mode-profile", args.restore_mode_profile]
                    )
            complete_existing_trial = (
                metrics_path.exists() and summary_path.exists()
            )
            partial_existing_trial = (
                metrics_path.exists() != summary_path.exists()
            )
            if partial_existing_trial and args.resume:
                parser.error(
                    f"partial trial cannot be resumed safely: {stem}"
                )
            if args.aggregate_only:
                if not metrics_path.exists() or not summary_path.exists():
                    parser.error(
                        f"missing existing trial files for {method} trial {trial_number}"
                    )
            elif args.resume and complete_existing_trial:
                print(f"reusing complete {method} trial {trial_number}", flush=True)
            else:
                print(f"running {method} trial {trial_number}", flush=True)
                with log_path.open("w", encoding="utf-8") as log:
                    subprocess.run(
                        command,
                        cwd=REPO_ROOT,
                        env=environment,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                        timeout=args.trial_timeout_seconds,
                    )
            trial_result = trial_metrics(
                metrics_path, summary_path, method == "stock"
            )
            pooled_initial_timings.extend(
                trial_result.pop("_initial_request_timings")
            )
            pooled_replay_timings.extend(
                trial_result.pop("_replay_request_timings")
            )
            trial_result["configured_error_bound"] = method_error_bound(
                method, args.error_bound
            )
            trials.append(trial_result)
        metric_keys = [
            "compression_ratio",
            "pcie_contender_gbps",
            "gpu_to_cpu_ms",
            "cpu_to_gpu_ms",
            "token_match_rate",
            "exact_match_rate",
            "initial_e2e_ms",
            "initial_task_accuracy",
            "replay_task_accuracy",
            "replay_e2e_ms",
            "replay_ttft_ms",
            "replay_tpot_ms",
            "initial_wall_ms",
            "replay_wall_ms",
            "initial_requests_per_second",
            "replay_requests_per_second",
            "initial_output_tokens_per_second",
            "replay_output_tokens_per_second",
            "restore_cpu_decode_ms",
            "restore_h2d_ms",
            "restore_gpu_decode_ms",
            "restore_scatter_ms",
            "restore_profiled_total_ms",
            "restore_h2d_fraction",
            "effective_h2d_gbps",
            "mean_restore_original_bytes",
            "mean_restore_transferred_bytes",
            "restore_decompression_gbps",
        ]
        all_results[method] = {
            key: describe([trial[key] for trial in trials])
            for key in metric_keys
            if key in trials[0]
        }
        all_results[method]["configured_error_bound"] = trials[0][
            "configured_error_bound"
        ]
        all_results[method]["request_latency_distribution"] = {
            "initial": summarize_request_timings(
                pooled_initial_timings,
                slo_ttft_ms=args.slo_ttft_ms,
                slo_e2e_ms=args.slo_e2e_ms,
            ),
            "replay": summarize_request_timings(
                pooled_replay_timings,
                slo_ttft_ms=args.slo_ttft_ms,
                slo_e2e_ms=args.slo_e2e_ms,
            ),
        }
        all_results[method]["trial_details"] = trials
    paired_comparisons = {}
    if args.trial_order == "interleaved" and len(args.methods) > 1:
        baseline_method = args.methods[0]
        baseline_trials = all_results[baseline_method]["trial_details"]
        for candidate_method in args.methods[1:]:
            candidate_trials = all_results[candidate_method]["trial_details"]
            paired_comparisons[candidate_method] = {
                "baseline": baseline_method,
                "candidate": candidate_method,
                "difference_definition": "candidate_minus_baseline",
                "metrics": paired_descriptions(
                    baseline_trials,
                    candidate_trials,
                    metric_keys,
                ),
            }

    output = {
        "schema_version": 1,
        "provenance": collect_runtime_provenance(REPO_ROOT, args.model),
        "model": args.model,
        "num_trials": args.trials,
        "warmup_offload": True,
        "trial_order": args.trial_order,
        "error_bound": args.error_bound,
        "error_bound_semantics": "default_for_unlabelled_cuszp_methods",
        "method_error_bounds": {
            method: method_error_bound(method, args.error_bound)
            for method in args.methods
        },
        "cuszp_mode": args.cuszp_mode,
        "adaptive_cuszp_modes": list(args.adaptive_cuszp_modes),
        "adaptive_candidates": list(args.adaptive_candidates),
        "restore_mode_profile": args.restore_mode_profile,
        "max_model_len": args.max_model_len,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "prompt_repeats": list(args.prompt_repeats),
        "cpu_offload_gb": args.cpu_offload_gb,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_tokens": args.max_tokens,
        "burst_slo_thresholds_ms": {
            "ttft": args.slo_ttft_ms,
            "e2e": args.slo_e2e_ms,
        },
        "prompt_style": args.prompt_style,
        "batch_prompts": args.batch_prompts,
        "profile_restore_stages": args.profile_restore_stages,
        "prompt_file": args.prompt_file,
        "batch_restore_transfers": args.batch_restore_transfers,
        "pcie_contender_mib": args.pcie_contender_mib,
        "pcie_contender_idle_us": args.pcie_contender_idle_us,
        "cost_aware_restore": args.cost_aware_restore,
        "prompt_batch_sizes": args.prompt_batch_sizes,
        "interarrival_ms": args.interarrival_ms,
        "methods": all_results,
        "paired_comparisons": paired_comparisons,
    }
    if args.quality_gate and not args.internal_single:
        output["quality_gate"] = evaluate_quality_gate(
            all_results,
            baseline_method=(args.quality_baseline_method or args.methods[0]),
            min_token_match_rate=args.quality_min_token_match_rate,
            min_exact_match_rate=args.quality_min_exact_match_rate,
            max_token_match_drop=args.quality_max_token_match_drop,
            max_exact_match_drop=args.quality_max_exact_match_drop,
            min_task_accuracy=args.quality_min_task_accuracy,
            max_task_accuracy_drop=args.quality_max_task_accuracy_drop,
        )
    else:
        output["quality_gate"] = {"enabled": False}
    output_path = out_dir / "aggregate.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2), flush=True)
    if (
        args.quality_gate
        and not args.internal_single
        and not output["quality_gate"]["passed"]
    ):
        print(
            f"quality gate failed; evidence retained in {output_path}",
            file=sys.stderr,
        )
        raise SystemExit(2)


if __name__ == "__main__":
    main()
