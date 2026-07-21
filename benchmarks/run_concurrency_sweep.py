"""Run a real-request concurrency sweep for the INFOCOM evaluation."""

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
DEFAULT_CONCURRENCY = (2, 4, 8, 16)

def runner_out_dir(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)



def total_handler_trial(trial: dict) -> float:
    return float(trial["gpu_to_cpu_ms"]) + float(trial["cpu_to_gpu_ms"])


def summarize_aggregate(document: dict) -> dict:
    methods = {}
    raw_trials = document["methods"]["async_raw"]["trial_details"]
    for method, metrics in document["methods"].items():
        trials = metrics["trial_details"]
        result = {
            "compression_ratio": metrics["compression_ratio"]["mean"],
            "gpu_to_cpu_ms": metrics["gpu_to_cpu_ms"]["mean"],
            "cpu_to_gpu_ms": metrics["cpu_to_gpu_ms"]["mean"],
            "total_handler_ms": (
                metrics["gpu_to_cpu_ms"]["mean"]
                + metrics["cpu_to_gpu_ms"]["mean"]
            ),
            "initial_e2e_ms": metrics["initial_e2e_ms"]["mean"],
            "replay_e2e_ms": metrics["replay_e2e_ms"]["mean"],
            "replay_ttft_ms": metrics["replay_ttft_ms"]["mean"],
            "token_match_rate": metrics["token_match_rate"]["mean"],
            "exact_match_rate": metrics["exact_match_rate"]["mean"],
            "quality_passed": document.get("quality_gate", {})
            .get("methods", {})
            .get(method, {})
            .get("passed"),
        }
        if "request_latency_distribution" in metrics:
            result["request_latency_distribution"] = metrics[
                "request_latency_distribution"
            ]
        for key in (
            "initial_wall_ms",
            "replay_wall_ms",
            "initial_requests_per_second",
            "replay_requests_per_second",
            "initial_output_tokens_per_second",
            "replay_output_tokens_per_second",
        ):
            if key in metrics:
                result[key] = metrics[key]["mean"]
        if method != "async_raw":
            differences = [
                total_handler_trial(candidate) - total_handler_trial(baseline)
                for baseline, candidate in zip(raw_trials, trials)
            ]
            result["total_handler_vs_raw_ms"] = describe(differences)
            result["paired_vs_raw"] = document.get(
                "paired_comparisons", {}
            ).get(method, {}).get("metrics", {})
        methods[method] = result
    return {
        "num_trials": document["num_trials"],
        "model": document["model"],
        "prompt_count": len(document["prompt_repeats"]),
        "prompt_repeat": document["prompt_repeats"][0],
        "burst_slo_thresholds_ms": document.get(
            "burst_slo_thresholds_ms", {}
        ),
        "methods": methods,
        "quality_gate_passed": document.get("quality_gate", {}).get("passed"),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Real-request concurrency sweep",
        "",
        "No synthetic PCIe copy contender is used. Every level submits unique",
        "long-context requests in one vLLM batch. V1 model runner is forced",
        "because vLLM marks UVA unavailable under WSL.",
        "",
        "| Concurrency | Method | Ratio | G2C ms | C2G ms | Total ms | "
        "Initial req/s | Replay req/s | Replay TTFT ms | Quality |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for level in summary["levels"]:
        if level.get("status") != "complete":
            lines.append(
                f"| {level['concurrency']} | {level['status']} | - | - | "
                "- | - | - | - | - | - |"
            )
            continue
        for method, metrics in level["result"]["methods"].items():
            quality = metrics.get("quality_passed")
            quality_text = (
                "pass" if quality is True
                else "reject" if quality is False
                else "n/a"
            )
            lines.append(
                "| {concurrency} | {method} | {ratio:.5f}x | {g2c:.3f} | "
                "{c2g:.3f} | {total:.3f} | {initial_rps:.3f} | "
                "{replay_rps:.3f} | {ttft:.3f} | {quality} |".format(
                    concurrency=level["concurrency"],
                    method=method,
                    ratio=metrics["compression_ratio"],
                    g2c=metrics["gpu_to_cpu_ms"],
                    c2g=metrics["cpu_to_gpu_ms"],
                    total=metrics["total_handler_ms"],
                    initial_rps=metrics.get(
                        "initial_requests_per_second", float("nan")
                    ),
                    replay_rps=metrics.get(
                        "replay_requests_per_second", float("nan")
                    ),
                    ttft=metrics["replay_ttft_ms"],
                    quality=quality_text,
                )
            )
    lines.append("")
    lines.extend(
        [
            "## Pooled burst tail latency and SLO",
            "",
            "Tails pool all requests across the five trials before computing",
            "percentiles. These are simultaneous-batch burst results, not an",
            "open-loop arrival-rate experiment.",
            "",
            "| Concurrency | Method | Requests | TTFT p95 | TTFT p99 | "
            "E2E p95 | E2E p99 | TTFT SLO | E2E SLO |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for level in summary["levels"]:
        if level.get("status") != "complete":
            continue
        for method, metrics in level["result"]["methods"].items():
            distribution = metrics.get("request_latency_distribution", {}).get(
                "initial", {}
            )
            if not distribution.get("count"):
                continue
            slo = distribution.get("slo", {})
            ttft_slo = slo.get("ttft_ms", {}).get("attainment")
            e2e_slo = slo.get("e2e_ms", {}).get("attainment")
            lines.append(
                "| {concurrency} | {method} | {count} | {ttft_p95:.3f} | "
                "{ttft_p99:.3f} | {e2e_p95:.3f} | {e2e_p99:.3f} | "
                "{ttft_slo} | {e2e_slo} |".format(
                    concurrency=level["concurrency"],
                    method=method,
                    count=distribution["count"],
                    ttft_p95=distribution["ttft_ms"]["p95"],
                    ttft_p99=distribution["ttft_ms"]["p99"],
                    e2e_p95=distribution["e2e_ms"]["p95"],
                    e2e_p99=distribution["e2e_ms"]["p99"],
                    ttft_slo=(
                        f"{100 * ttft_slo:.1f}%" if ttft_slo is not None else "n/a"
                    ),
                    e2e_slo=(
                        f"{100 * e2e_slo:.1f}%" if e2e_slo is not None else "n/a"
                    ),
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
    levels: list[dict],
) -> None:
    summary = {
        "schema_version": 1,
        "protocol": "real_request_concurrency_sweep",
        "model": model,
        "methods": methods,
        "trials": trials,
        "runner_environment": {"VLLM_USE_V2_MODEL_RUNNER": "0"},
        "synthetic_pcie_contender": False,
        "levels": levels,
    }
    (out_root / "concurrency_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_root / "CONCURRENCY_SUMMARY.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def upsert_level(levels: list[dict], result: dict) -> None:
    for index, existing in enumerate(levels):
        if existing.get("concurrency") == result.get("concurrency"):
            levels[index] = result
            return
    levels.append(result)
    levels.sort(key=lambda item: item["concurrency"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument(
        "--concurrency", type=int, nargs="+", default=DEFAULT_CONCURRENCY
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--out-root", default="data/qwen1.5b_real_concurrency_v3"
    )
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--repeat-upper-bound", type=int, default=260)
    parser.add_argument(
        "--kv-cache-memory-bytes", type=int, default=134217728
    )
    parser.add_argument("--cpu-offload-gb", type=float, default=4.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=8)
    parser.add_argument("--slo-ttft-ms", type=float, default=2000.0)
    parser.add_argument("--slo-e2e-ms", type=float, default=2500.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.trials < 2:
        parser.error("at least two trials are required")
    if any(level < 2 for level in args.concurrency):
        parser.error("concurrency levels must be at least two")
    if len(set(args.concurrency)) != len(args.concurrency):
        parser.error("concurrency levels must be unique")
    if not args.methods or args.methods[0] != "async_raw":
        parser.error("the first method must be async_raw")
    if any(not method.startswith("async_") for method in args.methods):
        parser.error("all methods must use the asynchronous worker")

    out_root = (REPO_ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    summary_path = out_root / "concurrency_summary.json"
    levels = []
    if args.resume and summary_path.is_file():
        existing = json.loads(summary_path.read_text(encoding="utf-8"))
        if existing.get("methods") != list(args.methods):
            parser.error("existing summary uses a different method list")
        levels = list(existing.get("levels", []))

    tokenizer = None
    if not args.aggregate_only:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            local_files_only=True,
        )
        common_repeat = calibrated_repeat(
            tokenizer,
            max_model_len=args.max_model_len,
            max_tokens=args.max_tokens,
            upper_bound=args.repeat_upper_bound,
            prompt_count=max(args.concurrency),
            style="disjoint",
        )
    else:
        common_repeat = None

    quality_rejections = False
    for concurrency in sorted(args.concurrency):
        level_out = out_root / f"c{concurrency}"
        aggregate_path = level_out / "aggregate.json"
        if args.aggregate_only:
            if not aggregate_path.is_file():
                parser.error(f"missing aggregate for concurrency {concurrency}")
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
                "--prompt-repeats", *([str(safe_repeat)] * concurrency),
                "--prompt-style", "disjoint",
                "--batch-prompts",
                "--batch-restore-transfers",
                "--error-bound", "1e-4",
                "--cuszp-mode", "fixed",
                "--cpu-offload-gb", str(args.cpu_offload_gb),
                "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                "--max-tokens", str(args.max_tokens),
                "--slo-ttft-ms", str(args.slo_ttft_ms),
                "--slo-e2e-ms", str(args.slo_e2e_ms),
                "--trial-timeout-seconds", "600",
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
                    "concurrency": concurrency,
                    "status": "dry_run",
                    "prompt_repeat": safe_repeat,
                    "command": command,
                })
                write_summary(
                    out_root,
                    model=args.model,
                    methods=list(args.methods),
                    trials=args.trials,
                    levels=levels,
                )
                continue
            runner_env = os.environ.copy()
            runner_env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
            completed = subprocess.run(
                command, cwd=REPO_ROOT, env=runner_env, check=False
            )
            if completed.returncode not in (0, 2):
                upsert_level(levels, {
                    "concurrency": concurrency,
                    "status": "runner_failed",
                    "returncode": completed.returncode,
                    "prompt_repeat": safe_repeat,
                })
                write_summary(
                    out_root,
                    model=args.model,
                    methods=list(args.methods),
                    trials=args.trials,
                    levels=levels,
                )
                return completed.returncode or 1
            if not aggregate_path.is_file():
                raise FileNotFoundError(aggregate_path)
            document = json.loads(aggregate_path.read_text(encoding="utf-8"))

        result = summarize_aggregate(document)
        quality_rejections |= result["quality_gate_passed"] is False
        upsert_level(levels, {
            "concurrency": concurrency,
            "status": "complete",
            "aggregate": str(aggregate_path.relative_to(REPO_ROOT)),
            "prompt_repeat": safe_repeat,
            "result": result,
        })
        write_summary(
            out_root,
            model=args.model,
            methods=list(args.methods),
            trials=args.trials,
            levels=levels,
        )

    if quality_rejections:
        print("Concurrency sweep completed with quality rejection(s).")
    print(out_root / "CONCURRENCY_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
