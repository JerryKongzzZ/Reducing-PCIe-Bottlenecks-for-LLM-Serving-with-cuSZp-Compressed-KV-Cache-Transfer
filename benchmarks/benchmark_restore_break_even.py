"""Calibrate CPU-to-GPU KV restore stages and compression break-even bandwidth."""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalKVCacheTensor,
)

from integration.compression_pipeline.vllm_v1_compressed_offload import (
    CompressedCpuGpuOffloadingHandlers,
    RestoreStageTimings,
)


def describe(values):
    return {
        "mean": statistics.mean(values),
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "trials": list(values),
    }


def make_caches(num_pages, page_bytes):
    if page_bytes % 2:
        raise ValueError("BF16 page size must be divisible by two")
    page_numel = page_bytes // 2
    backing = torch.randn(
        2 * num_pages, page_numel, dtype=torch.bfloat16, device="cuda"
    )
    byte_view = backing.view(torch.uint8).view(torch.int8).reshape(2 * num_pages, -1)
    caches = CanonicalKVCaches(
        tensors=[
            CanonicalKVCacheTensor(tensor=byte_view, page_size_bytes=page_bytes)
        ],
        group_data_refs=[
            [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=page_bytes)]
        ],
    )
    return backing, caches


def measure_codec(caches, codec, error_bound, num_pages, trials, cuszp_mode, batch_restore):
    handlers = CompressedCpuGpuOffloadingHandlers(
        kv_caches=caches,
        block_size_factor=1,
        error_bound=error_bound,
        device_id=0,
        metrics_path=None,
        codec=codec,
        profile_restore_stages=True,
        batch_restore_transfers=batch_restore,
        cuszp_mode=cuszp_mode,
    )
    bundles = [
        handlers.gpu_to_cpu_handler._compress_bundle((0,), page_id)
        for page_id in range(num_pages)
    ]
    original_bytes = sum(bundle.original_bytes for bundle in bundles)
    transferred_bytes = sum(bundle.compressed_size for bundle in bundles)

    elapsed_values = []
    stage_trials = []
    for trial_idx in range(trials + 1):
        for page_id in range(num_pages, 2 * num_pages):
            caches.tensors[0].tensor[page_id].zero_()
        torch.cuda.synchronize()
        started = time.perf_counter()
        stages = RestoreStageTimings()
        restore_items = list(enumerate(bundles, start=num_pages))
        if batch_restore:
            stages = handlers.cpu_to_gpu_handler._decompress_bundles_batched(
                restore_items
            )
        else:
            for page_id, bundle in restore_items:
                stages.add(
                    handlers.cpu_to_gpu_handler._decompress_bundle(page_id, bundle)
                )
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        if trial_idx > 0:
            elapsed_values.append(elapsed)
            stage_trials.append(stages)
    page_values = (
        caches.tensors[0]
        .tensor.view(torch.uint8)
        .view(torch.bfloat16)
        .reshape(2 * num_pages, -1)
    )
    restored_values = page_values[num_pages:].to(torch.float32)
    reference_values = page_values[:num_pages].to(torch.float32)
    finite = bool(torch.isfinite(restored_values).all().item())
    max_abs_error = float(
        (reference_values - restored_values).abs().max().item()
    )


    stage_names = (
        "cpu_decode_seconds",
        "h2d_seconds",
        "gpu_decode_seconds",
        "scatter_seconds",
    )
    stage_summary = {
        name: describe([getattr(item, name) for item in stage_trials])
        for name in stage_names
    }
    mean_h2d = stage_summary["h2d_seconds"]["mean"]
    return {
        "codec": codec,
        "error_bound": error_bound,
        "original_bytes": original_bytes,
        "finite": finite,
        "max_abs_error": max_abs_error,
        "transferred_bytes": transferred_bytes,
        "compression_ratio": original_bytes / transferred_bytes,
        "elapsed_seconds": describe(elapsed_values),
        "restore_stages": stage_summary,
        "effective_h2d_gbps": (
            transferred_bytes * 8.0 / mean_h2d / 1e9 if mean_h2d > 0 else None
        ),
    }


def add_break_even(raw, compressed):
    raw_non_h2d = (
        raw["restore_stages"]["cpu_decode_seconds"]["mean"]
        + raw["restore_stages"]["gpu_decode_seconds"]["mean"]
        + raw["restore_stages"]["scatter_seconds"]["mean"]
    )
    compressed_non_h2d = (
        compressed["restore_stages"]["cpu_decode_seconds"]["mean"]
        + compressed["restore_stages"]["gpu_decode_seconds"]["mean"]
        + compressed["restore_stages"]["scatter_seconds"]["mean"]
    )
    extra_seconds = max(compressed_non_h2d - raw_non_h2d, 0.0)
    saved_bytes = raw["transferred_bytes"] - compressed["transferred_bytes"]
    compressed["incremental_non_h2d_seconds"] = extra_seconds
    compressed["saved_transfer_bytes"] = saved_bytes
    compressed["break_even_h2d_gbps"] = (
        saved_bytes * 8.0 / extra_seconds / 1e9
        if saved_bytes > 0 and extra_seconds > 0
        else None
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--page-mib", type=float, nargs="+", default=(1, 4, 16))
    parser.add_argument("--batch-pages", type=int, nargs="+", default=(1, 4, 8))
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--error-bound", type=float, default=1e-5)
    parser.add_argument("--codecs", nargs="+", default=("raw", "cuszp"))
    parser.add_argument(
        "--cuszp-mode", choices=("fixed", "plain", "outlier"), default="fixed"
    )
    parser.add_argument("--batch-restore-transfers", action="store_true")
    parser.add_argument(
        "--output", default="data/restore_break_even/rtx5080_calibration.json"
    )
    args = parser.parse_args()
    if args.trials < 1 or any(value <= 0 for value in args.page_mib):
        parser.error("trials and page sizes must be positive")
    if any(value <= 0 for value in args.batch_pages):
        parser.error("batch page counts must be positive")
    if "raw" not in args.codecs:
        parser.error("--codecs must include raw for break-even accounting")

    torch.manual_seed(0)
    results = []
    for page_mib in args.page_mib:
        page_bytes = int(page_mib * (1 << 20))
        for num_pages in args.batch_pages:
            _backing, caches = make_caches(num_pages, page_bytes)
            methods = {}
            for codec in args.codecs:
                methods[codec] = measure_codec(
                    caches,
                    codec,
                    args.error_bound,
                    num_pages,
                    args.trials,
                    args.cuszp_mode,
                    args.batch_restore_transfers,
                )
            for codec, result in methods.items():
                if codec != "raw":
                    add_break_even(methods["raw"], result)
            results.append({
                "page_mib": page_mib,
                "batch_pages": num_pages,
                "batch_original_mib": page_mib * num_pages,
                "methods": methods,
            })
            del caches, _backing
            torch.cuda.empty_cache()

    output = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(0),
        "error_bound": args.error_bound,
        "trials": args.trials,
        "cuszp_mode": args.cuszp_mode,
        "batch_restore_transfers": args.batch_restore_transfers,
        "results": results,
    }
    output_path = (REPO_ROOT / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
