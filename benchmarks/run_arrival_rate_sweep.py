"""Run fixed-rate open-loop vLLM workloads for the INFOCOM evaluation."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.prompt_workload import calibrated_repeat
from benchmarks.run_vllm_repeated_smoke import describe


REPEATED_RUNNER = REPO_ROOT / "benchmarks" / "run_vllm_repeated_smoke.py"
DEFAULT_METHODS = ("async_raw", "async_cuszp_1e-5")
DEFAULT_RATES = (2.0, 4.0, 6.0)


def runner_out_dir(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def rate_slug(rate: float) -> str:
    return f"r{rate:g}".replace(".", "p")


def total_handler_trial(trial: dict) -> float:
    return float(trial["gpu_to_cpu_ms"]) + float(trial["cpu_to_gpu_ms"])


def summarize_aggregate(document: dict) -> dict:
    raw_trials = document["methods"]["async_raw"]["trial_details"]
    methods = {}
    for method, metrics in document["methods"].items():
        distribution = metrics["request_latency_distribution"]["initial"]
        result = {
            "compression_ratio": metrics["compression_ratio"]["mean"],
            "total_handler_ms": (
                metrics["gpu_to_cpu_ms"]["mean"]
                + metrics["cpu_to_gpu_ms"]["mean"]
            ),
            "achieved_requests_per_second": metrics[
                "initial_requests_per_second"
            ]["mean"],
            "latency": distribution,
            "quality_passed": document.get("quality_gate", {})
            .get("methods", {})
            .get(method, {})
            .get("passed"),
        }
        if method != "async_raw":
            trials = metrics["trial_details"]
            result["total_handler_vs_raw_ms"] = describe(
                [
                    total_handler_trial(candidate)
                    - total_handler_trial(baseline)
                    for baseline, candidate in zip(raw_trials, trials)
                ]
            )
            result["paired_vs_raw"] = document.get(
                "paired_comparisons", {}
            ).get(method, {}).get("metrics", {})
        methods[method] = result
    return {
        "model": document["model"],
        "num_trials": document["num_trials"],
        "request_count": len(document["prompt_repeats"]),
        "prompt_repeat": document["prompt_repeats"][0],
        "interarrival_ms": document["interarrival_ms"],
        "offered_requests_per_second": 1000.0 / document["interarrival_ms"],
        "burst_slo_thresholds_ms": document["burst_slo_thresholds_ms"],
        "quality_gate_passed": document.get("quality_gate", {}).get("passed"),
        "methods": methods,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Fixed-rate open-loop arrival sweep",
        "",
        "Requests are injected at fixed wall-clock arrival times through",
        "LLMEngine.add_request. This is an open-loop workload; overdue arrivals",
        "retain their scheduled arrival timestamp so queueing is included in TTFT.",
        "",
        "| Offered req/s | Method | Achieved req/s | Handler ms | TTFT p95 | "
        "TTFT p99 | E2E p95 | E2E p99 | TTFT SLO | E2E SLO | Quality |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for level in summary["levels"]:
        if level.get("status") != "complete":
            continue
        offered = level["result"]["offered_requests_per_second"]
        for method, metrics in level["result"]["methods"].items():
            latency = metrics["latency"]
            slo = latency.get("slo", {})
            ttft_slo = slo.get("ttft_ms", {}).get("attainment")
            e2e_slo = slo.get("e2e_ms", {}).get("attainment")
            quality = metrics.get("quality_passed")
            lines.append(
                "| {offered:.3f} | {method} | {achieved:.3f} | "
                "{handler:.3f} | {ttft95:.3f} | {ttft99:.3f} | "
                "{e2e95:.3f} | {e2e99:.3f} | {ttft_slo} | {e2e_slo} | "
                "{quality} |".format(
                    offered=offered,
                    method=method,
                    achieved=metrics["achieved_requests_per_second"],
                    handler=metrics["total_handler_ms"],
                    ttft95=latency["ttft_ms"]["p95"],
                    ttft99=latency["ttft_ms"]["p99"],
                    e2e95=latency["e2e_ms"]["p95"],
                    e2e99=latency["e2e_ms"]["p99"],
                    ttft_slo=(
                        f"{100 * ttft_slo:.1f}%" if ttft_slo is not None else "n/a"
                    ),
                    e2e_slo=(
                        f"{100 * e2e_slo:.1f}%" if e2e_slo is not None else "n/a"
                    ),
                    quality=(
                        "pass" if quality is True
                        else "reject" if quality is False
                        else "n/a"
                    ),
                )
            )
    lines.append("")
    lines.extend(
        [
            f"## {summary['trials']}-trial paired differences",
            "",
            "Differences are cuSZp minus raw; intervals are paired 95% CI",
            "half-widths. Negative latency and positive throughput are better.",
            "",
            "| Offered req/s | Handler diff ms | Achieved req/s diff | "
            "Mean initial E2E diff ms |",
            "|---:|---:|---:|---:|",
        ]
    )
    for level in summary["levels"]:
        if level.get("status") != "complete":
            continue
        result = level["result"]
        candidate = result["methods"].get("async_cuszp_1e-5")
        if not candidate or "paired_vs_raw" not in candidate:
            continue
        handler = candidate["total_handler_vs_raw_ms"]
        throughput = candidate["paired_vs_raw"][
            "initial_requests_per_second"
        ]
        e2e = candidate["paired_vs_raw"]["initial_e2e_ms"]
        lines.append(
            "| {rate:.3f} | {handler:.3f} +/- {handler_ci:.3f} | "
            "{throughput:.3f} +/- {throughput_ci:.3f} | "
            "{e2e:.3f} +/- {e2e_ci:.3f} |".format(
                rate=result["offered_requests_per_second"],
                handler=handler["mean"],
                handler_ci=handler["ci95_half_width"],
                throughput=throughput["mean"],
                throughput_ci=throughput["ci95_half_width"],
                e2e=e2e["mean"],
                e2e_ci=e2e["ci95_half_width"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def write_summary(
    out_root: Path,
    *,
    model: str,
    methods: list[str],
    trials: int,
    request_count: int,
    levels: list[dict],
) -> None:
    summary = {
        "schema_version": 1,
        "protocol": "fixed_rate_open_loop",
        "model": model,
        "methods": methods,
        "trials": trials,
        "request_count": request_count,
        "runner_environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
        "levels": levels,
    }
    (out_root / "arrival_rate_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_root / "ARRIVAL_RATE_SUMMARY.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def upsert_level(levels: list[dict], result: dict) -> None:
    for index, existing in enumerate(levels):
        if existing.get("offered_requests_per_second") == result.get(
            "offered_requests_per_second"
        ):
            levels[index] = result
            return
    levels.append(result)
    levels.sort(key=lambda item: item["offered_requests_per_second"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--rates", type=float, nargs="+", default=DEFAULT_RATES)
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--request-count", type=int, default=16)
    parser.add_argument(
        "--out-root", default="data/qwen1.5b_open_loop_arrival_v1"
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--repeat-upper-bound", type=int, default=260)
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=134217728
    )
    parser.add_argument("--cpu-offload-gb", type=float, default=4.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--slo-ttft-ms", type=float, default=500.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=750.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("at least two trials are required")
    if args.request_count < 2:
        parser.error("request count must be at least two")
    if any(rate <= 0 for rate in args.rates):
        parser.error("arrival rates must be positive")
    if len(set(args.rates)) != len(args.rates):
        parser.error("arrival rates must be unique")
    if not args.methods or args.methods[0] != "async_raw":
        parser.error("the first method must be async_raw")

    out_root = (REPO_ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "arrival_rate_summary.json"
    levels = []
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        levels = list(existing.get("levels", []))

    common_repeat = None
    if not args.aggregate_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model, local_files_only=True
        )
        common_repeat = calibrated_repeat(
            tokenizer,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            upper_bound=args.repeat_upper_bound,
            prompt_count=args.request_count,
            style="disjoint",
        )

    quality_rejections = False
    for rate in sorted(args.rates):
        level_out = out_root / rate_slug(rate)
        aggregate_path = level_out / "aggregate.json"
        if args.aggregate_only:
            if not aggregate_path.is_file():
                parser.error(f"missing aggregate for rate {rate:g}")
            document = json.loads(aggregate_path.read_text(encoding="utf-8"))
            safe_repeat = int(document["prompt_repeats"][0])
        else:
            safe_repeat = common_repeat
            command = [
                sys.executable,
                str(REPEATED_RUNNER),
                "--methods", *args.methods,
                "--trials", str(args.trials),
                "--trial-order", "interleaved",
                "--out-dir", runner_out_dir(level_out),
                "--model", args.model,
                "--max-model-len", str(args.max_model_len),
                "--kv-cache-memory-bytes", str(args.kv_cache_memory_bytes),
                "--prompt-repeats", *([str(safe_repeat)] * args.request_count),
                "--prompt-style", "disjoint",
                "--interarrival-ms", str(1000.0 / rate),
                "--batch-restore-transfers",
                "--error-bound", "1e-4",
                "--cuszp-mode", "fixed",
                "--cpu-offload-gb", str(args.cpu_offload_gb),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-tokens", str(args.max_tokens),
                "--slo-ttft-ms", str(args.slo_ttft_ms),
                "--slo-e2e-ms", str(args.slo_e2e_ms),
                "--trial-timeout-seconds", "900",
                "--quality-gate",
                "--quality-baseline-method", "async_raw",
                "--quality-max-token-match-drop", "0",
                "--quality-max-exact-match-drop", "0",
            ]
            if args.resume:
                command.append("--resume")
            print(shlex.join(command), flush=True)
            if args.dry_run:
                upsert_level(levels, {
                    "offered_requests_per_second": rate,
                    "status": "dry_run",
                    "command": command,
                })
                write_summary(
                    out_root,
                    model=args.model,
                    methods=list(args.methods),
                    trials=args.trials,
                    request_count=args.request_count,
                    levels=levels,
                )
                continue
            environment = os.environ.copy()
            environment["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=environment, check=False
            )
            if completed.returncode not in (0, 2):
                return completed.returncode or 1
            document = json.loads(aggregate_path.read_text(encoding="utf-8"))

        result = summarize_aggregate(document)
        quality_rejections |= result["quality_gate_passed"] is False
        upsert_level(levels, {
            "offered_requests_per_second": rate,
            "status": "complete",
            "aggregate": str(aggregate_path.relative_to(REPO_ROOT)),
            "result": result,
        })
        write_summary(
            out_root,
            model=args.model,
            methods=list(args.methods),
            trials=args.trials,
            request_count=args.request_count,
            levels=levels,
        )

    if quality_rejections:
        print("Arrival sweep completed with quality rejection(s).")
    print(out_root / "ARRIVAL_RATE_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
