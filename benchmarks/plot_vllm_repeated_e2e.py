"""Plot the canonical five-trial vLLM comparison without synthetic fallbacks."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/vllm_repeated_e2e/aggregate.json")
    parser.add_argument(
        "--output", default="data/figures/vllm_repeated_e2e_comparison.png"
    )
    args = parser.parse_args()
    source = Path(args.input)
    if not source.exists():
        raise FileNotFoundError("measured aggregate is required; no fallback data")
    data = json.loads(source.read_text(encoding="utf-8"))
    methods = data["methods"]
    preferred_order = [
        "stock", "raw", "cuszp", "async_cuszp_1e-5", "async_cuszp",
        "async_cuszp_1e-3", "int8", "zlib", "zstd", "async_zstd", "lz4",
        "async_lz4", "adaptive", "async_adaptive"
    ]
    order = [name for name in preferred_order if name in methods]
    label_map = {
        "stock": "Stock vLLM",
        "raw": "Sync raw",
        "cuszp": "cuSZp",
        "async_cuszp": "Async cuSZp",
        "async_cuszp_1e-5": "Async cuSZp 1e-5",
        "async_cuszp_1e-3": "Async cuSZp 1e-3",
        "int8": "INT8",
        "zlib": "zlib",
        "zstd": "zstd",
        "async_zstd": "Async zstd",
        "lz4": "LZ4",
        "async_lz4": "Async LZ4",
        "adaptive": "Adaptive",
        "async_adaptive": "Async adaptive",
    }
    labels = [label_map[name] for name in order]

    ratios = [methods[name]["compression_ratio"]["mean"] for name in order]
    latency = [methods[name]["initial_e2e_ms"]["mean"] for name in order]
    latency_ci = [
        methods[name]["initial_e2e_ms"]["ci95_half_width"] for name in order
    ]
    exact = [methods[name]["exact_match_rate"]["mean"] for name in order]
    color_map = {
        "stock": "#4d4d4d",
        "raw": "#7f8c8d",
        "cuszp": "#2878b5",
        "async_cuszp": "#3a9ad9",
        "async_cuszp_1e-5": "#73b7df",
        "async_cuszp_1e-3": "#2468a2",
        "int8": "#f39c12",
        "zlib": "#8e6c8a",
        "zstd": "#7a5195",
        "async_zstd": "#9856a3",
        "lz4": "#ef8354",
        "async_lz4": "#f6a15e",
        "adaptive": "#2a9d8f",
        "async_adaptive": "#e07a1f",
    }
    colors = [color_map[name] for name in order]
    x = np.arange(len(order))

    figure_width = max(9.2, 2.15 * len(order) + 1.5)
    fig, axes = plt.subplots(1, 2, figsize=(figure_width, 4.2))
    axes[0].bar(x, ratios, color=colors)
    axes[0].axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_ylabel("Transfer compression ratio (×)")
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    ratio_ceiling = max(ratios) * 1.25
    axes[0].set_ylim(0, ratio_ceiling)
    for idx, value in enumerate(ratios):
        axes[0].text(
            idx, value + ratio_ceiling * 0.018, f"{value:.2f}×",
            ha="center", fontsize=8
        )

    axes[1].bar(x, latency, yerr=latency_ci, capsize=4, color=colors)
    axes[1].set_ylabel("Initial request E2E latency (ms)")
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    latency_ceiling = (
        max(value + error for value, error in zip(latency, latency_ci)) * 1.25
    )
    axes[1].set_ylim(0, latency_ceiling)
    for idx, (value, quality) in enumerate(zip(latency, exact)):
        axes[1].text(
            idx,
            value + latency_ci[idx] + latency_ceiling * 0.025,
            f"exact {quality * 100:.0f}%",
            ha="center",
            fontsize=7,
        )

    model = data.get("model", "vLLM")
    num_trials = data.get("num_trials", "?")
    max_model_len = data.get("max_model_len")
    length_label = f", max length {max_model_len}" if max_model_len else ""
    fig.suptitle(
        f"{model} vLLM offload: {num_trials} isolated trials{length_label} (95% CI)"
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.91))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
