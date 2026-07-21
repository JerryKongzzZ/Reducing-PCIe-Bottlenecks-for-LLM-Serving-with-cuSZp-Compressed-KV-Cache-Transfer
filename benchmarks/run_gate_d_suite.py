"""Run the fair INFOCOM Gate D protocol across the eight cached models."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
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

REPEATED_RUNNER = REPO_ROOT / "benchmarks" / "run_vllm_repeated_smoke.py"
PAPER_PROTOCOL = "legacy_v1"
PROTOCOL_PROMPT_STYLES = {
    "legacy_v1": "legacy_disjoint_v1",
    "corrected_v2": "disjoint",
}
DEFAULT_METHODS = (
    "async_raw",
    "async_cuszp_1e-5",
    "async_int8",
    "async_zstd",
    "async_lz4",
)


@dataclass(frozen=True)
class ModelWorkload:
    model: str
    max_model_len: int
    prompt_repeat: int
    kv_cache_memory_bytes: int

    @property
    def slug(self) -> str:
        return self.model.replace("/", "_").replace("-", "_")

    @property
    def cache_dir_name(self) -> str:
        return "models--" + self.model.replace("/", "--")


MODEL_WORKLOADS = {
    workload.model: workload
    for workload in (
        ModelWorkload("gpt2", 1024, 65, 128 * 1024 * 1024),
        ModelWorkload("facebook/opt-125m", 2048, 130, 128 * 1024 * 1024),
        ModelWorkload("facebook/opt-350m", 2048, 130, 256 * 1024 * 1024),
        ModelWorkload("EleutherAI/pythia-160m", 2048, 130, 128 * 1024 * 1024),
        ModelWorkload("EleutherAI/pythia-410m", 2048, 130, 256 * 1024 * 1024),
        ModelWorkload(
            "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T",
            2048,
            130,
            128 * 1024 * 1024,
        ),
        ModelWorkload("Qwen/Qwen2.5-0.5B", 4096, 260, 128 * 1024 * 1024),
        ModelWorkload("Qwen/Qwen2.5-1.5B", 4096, 260, 128 * 1024 * 1024),
    )
}


def cache_path(workload: ModelWorkload, cache_root: Path) -> Path:
    return cache_root / workload.cache_dir_name


def runner_out_dir(path: Path) -> str:
    """Keep portable relative paths in-repo and allow absolute scratch roots."""
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def existing_workload_version(summary: dict) -> str:
    """Treat unversioned archived Gate D summaries as the published v1."""
    return summary.get("workload_version", PAPER_PROTOCOL)


def summarize_aggregate(document: dict) -> dict:
    methods = {}
    quality_methods = document.get("quality_gate", {}).get("methods", {})
    for method, metrics in document.get("methods", {}).items():
        methods[method] = {
            "compression_ratio": metrics["compression_ratio"]["mean"],
            "gpu_to_cpu_ms": metrics["gpu_to_cpu_ms"]["mean"],
            "cpu_to_gpu_ms": metrics["cpu_to_gpu_ms"]["mean"],
            "token_match_rate": metrics["token_match_rate"]["mean"],
            "exact_match_rate": metrics["exact_match_rate"]["mean"],
            "quality_passed": quality_methods.get(method, {}).get("passed"),
        }
    return {
        "model": document.get("model"),
        "num_trials": document.get("num_trials"),
        "quality_gate_passed": document.get("quality_gate", {}).get("passed"),
        "methods": methods,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Gate D multi-model summary",
        "",
        "All methods use the asynchronous offload worker, batched restore, and",
        "the same per-model prompt/KV-cache workload. `quality` is evaluated",
        "per trial relative to `async_raw`.",
        "",
        "| Model | Method | Ratio | G2C ms | C2G ms | Token | Exact | Quality |",
        "|---|---|---:|---:|---:|---:|---:|:---:|",
    ]
    for model_result in summary["models"]:
        if model_result.get("status") != "complete":
            lines.append(
                f"| {model_result['model']} | {model_result['status']} | - | - | - | - | - | - |"
            )
            continue
        for method, metrics in model_result["result"]["methods"].items():
            quality = metrics["quality_passed"]
            quality_text = "pass" if quality else "reject" if quality is False else "n/a"
            lines.append(
                "| {model} | {method} | {ratio:.5f}x | {g2c:.3f} | "
                "{c2g:.3f} | {token:.5f} | {exact:.5f} | {quality} |".format(
                    model=model_result["model"],
                    method=method,
                    ratio=metrics["compression_ratio"],
                    g2c=metrics["gpu_to_cpu_ms"],
                    c2g=metrics["cpu_to_gpu_ms"],
                    token=metrics["token_match_rate"],
                    exact=metrics["exact_match_rate"],
                    quality=quality_text,
                )
            )
    lines.append("")
    return "\n".join(lines)


def write_summary(
    out_root: Path,
    models: list[dict],
    methods: list[str],
    *,
    workload_version: str = PAPER_PROTOCOL,
) -> None:
    summary = {
        "schema_version": 1,
        "protocol": "infocom_gate_d_fair_async",
        "workload_version": workload_version,
        "prompt_style": PROTOCOL_PROMPT_STYLES[workload_version],
        "methods": methods,
        "runner_environment": {
            "VLLM_USE_V2_MODEL_RUNNER": "0",
        },
        "models": models,
    }
    (out_root / "gate_d_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (out_root / "GATE_D_SUMMARY.md").write_text(
        render_markdown(summary), encoding="utf-8"
    )


def upsert_model_result(models: list[dict], result: dict) -> None:
    for index, existing in enumerate(models):
        if existing.get("model") == result.get("model"):
            models[index] = result
            return
    models.append(result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_WORKLOADS),
        default=tuple(MODEL_WORKLOADS),
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--out-root", default="data/gate_d_fair_async_probe")
    parser.add_argument(
        "--workload-version",
        choices=tuple(PROTOCOL_PROMPT_STYLES),
        default=PAPER_PROTOCOL,
        help=(
            "legacy_v1 exactly reproduces the paper's six-seed/eight-slot "
            "workload; corrected_v2 creates eight unique request streams"
        ),
    )
    parser.add_argument(
        "--cache-root",
        default=str(Path.home() / ".cache" / "huggingface" / "hub"),
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--require-all-models", action="store_true")
    args = parser.parse_args()
    if args.trials < 2:
        parser.error("Gate D requires at least two trials; use five for paper tables")
    if not args.methods or args.methods[0] != "async_raw":
        parser.error("the first Gate D method must be async_raw")
    if any(not method.startswith("async_") for method in args.methods):
        parser.error("Gate D fair methods must all use the async worker")

    out_root = (REPO_ROOT / args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    prompt_style = PROTOCOL_PROMPT_STYLES[args.workload_version]
    cache_root = Path(args.cache_root).expanduser().resolve()
    summary_path = out_root / "gate_d_summary.json"
    model_results = []
    if args.resume and summary_path.is_file():
        existing_summary = json.loads(
            summary_path.read_text(encoding="utf-8")
        )
        if existing_summary.get("methods") != list(args.methods):
            parser.error(
                "existing Gate D summary uses a different method list"
            )
        if existing_workload_version(existing_summary) != args.workload_version:
            parser.error(
                "existing Gate D summary uses a different workload version; "
                "choose a new --out-root instead of relabeling old results"
            )
        model_results = list(existing_summary.get("models", []))
    quality_rejections = False

    for model_name in args.models:
        workload = MODEL_WORKLOADS[model_name]
        model_cache = cache_path(workload, cache_root)
        if not model_cache.is_dir():
            result = {
                "model": model_name,
                "status": "missing_model_cache",
                "cache_path": str(model_cache),
                "workload": asdict(workload),
            }
            upsert_model_result(model_results, result)
            write_summary(
                out_root,
                model_results,
                args.methods,
                workload_version=args.workload_version,
            )
            if args.require_all_models:
                raise FileNotFoundError(model_cache)
            continue

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            workload.model,
            cache_dir=cache_root,
            local_files_only=True,
        )
        safe_repeat = calibrated_repeat(
            tokenizer,
            max_model_len=workload.max_model_len,
            max_tokens=8,
            upper_bound=workload.prompt_repeat,
            prompt_count=8,
            style=prompt_style,
        )
        if safe_repeat != workload.prompt_repeat:
            print(
                f"calibrated {workload.model} prompt repeat: "
                f"{workload.prompt_repeat} -> {safe_repeat}",
                flush=True,
            )
            workload = replace(workload, prompt_repeat=safe_repeat)

        model_out = out_root / workload.slug
        command = [
            sys.executable,
            str(REPEATED_RUNNER),
            "--methods", *args.methods,
            "--trials", str(args.trials),
            "--trial-order", "interleaved",
            "--out-dir", runner_out_dir(model_out),
            "--model", workload.model,
            "--max-model-len", str(workload.max_model_len),
            "--kv-cache-memory-bytes", str(workload.kv_cache_memory_bytes),
            "--prompt-repeats", *([str(workload.prompt_repeat)] * 8),
            "--prompt-style", prompt_style,
            "--batch-prompts",
            "--batch-restore-transfers",
            "--error-bound", "1e-4",
            "--cuszp-mode", "fixed",
            "--cpu-offload-gb", "4",
            "--gpu-memory-utilization", "0.8",
            "--max-tokens", "8",
            "--trial-timeout-seconds", "300",
            "--quality-gate",
            "--quality-baseline-method", "async_raw",
            "--quality-max-token-match-drop", "0",
            "--quality-max-exact-match-drop", "0",
        ]
        if args.resume:
            command.append("--resume")
        print(shlex.join(command), flush=True)
        if args.dry_run:
            upsert_model_result(model_results, {
                "model": model_name,
                "status": "dry_run",
                "cache_path": str(model_cache),
                "workload": asdict(workload),
                "command": command,
            })
            write_summary(
                out_root,
                model_results,
                args.methods,
                workload_version=args.workload_version,
            )
            continue

        runner_env = os.environ.copy()
        runner_env["VLLM_USE_V2_MODEL_RUNNER"] = "0"
        completed = subprocess.run(
            command, cwd=REPO_ROOT, env=runner_env, check=False
        )
        aggregate_path = model_out / "aggregate.json"
        if completed.returncode not in (0, 2) or not aggregate_path.is_file():
            upsert_model_result(model_results, {
                "model": model_name,
                "status": "runner_failed",
                "returncode": completed.returncode,
                "workload": asdict(workload),
            })
            write_summary(
                out_root,
                model_results,
                args.methods,
                workload_version=args.workload_version,
            )
            return completed.returncode or 1
        document = json.loads(aggregate_path.read_text(encoding="utf-8"))
        result = summarize_aggregate(document)
        quality_rejections |= result["quality_gate_passed"] is False
        upsert_model_result(model_results, {
            "model": model_name,
            "status": "complete",
            "aggregate": str(aggregate_path.relative_to(REPO_ROOT)),
            "workload": asdict(workload),
            "result": result,
        })
        write_summary(
            out_root,
            model_results,
            args.methods,
            workload_version=args.workload_version,
        )

    if quality_rejections:
        print("Gate D completed with one or more quality rejections.")
    print(out_root / "GATE_D_SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
