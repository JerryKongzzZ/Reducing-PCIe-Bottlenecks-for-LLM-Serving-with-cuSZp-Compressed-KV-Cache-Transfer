"""
Plot evaluation summary JSON produced by `evaluate_policies.py`.

Produces PNG figures in `data/figures/`.
"""
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _build_policy_metrics(model_id, data):
    orig = _safe_float(data.get('orig_size_bytes', 0), 0.0)
    policies = ['baseline', 'static_cuszp', 'adaptive_red_sim', 'int8', 'zlib']

    labels = []
    swap_out_times = []
    swap_in_times = []
    ratios = []
    max_errors = []

    for policy in policies:
        entry = data.get(policy, {})
        if policy == 'baseline':
            d2h_time = _safe_float(entry.get('d2h_time'), 0.0)
            h2d_time = _safe_float(entry.get('h2d_time'), 0.0)
            comp_size = orig
            comp_time = 0.0
            decomp_time = 0.0
            max_error = None
        else:
            comp_size = _safe_float(entry.get('comp_size'), 0.0)
            comp_time = _safe_float(entry.get('comp_time'), 0.0)
            decomp_time = _safe_float(entry.get('decomp_time'), 0.0)
            d2h_time = _safe_float(entry.get('d2h_time'), 0.0)
            h2d_time = _safe_float(entry.get('h2d_time'), 0.0)
            max_error = entry.get('max_error', None)

        labels.append(policy.replace('_', ' ').replace('red sim', 'red'))
        swap_out_times.append(comp_time + (comp_size / (max(_safe_float(entry.get('d2h_bandwidth'), 0.0) * 1e9, 1e-12)) if policy == 'baseline' else 0.0))
        swap_in_times.append(decomp_time + (comp_size / (max(_safe_float(entry.get('h2d_bandwidth'), 0.0) * 1e9, 1e-12)) if policy == 'baseline' else 0.0))

        if comp_size > 0:
            ratios.append(orig / comp_size)
        else:
            ratios.append(0.0)

        if max_error is None:
            max_errors.append(None)
        else:
            max_errors.append(_safe_float(max_error, 0.0))

    # Use the effective transfer model from the evaluation summary when available.
    for policy in policies:
        entry = data.get(policy, {})
        if policy == 'baseline':
            continue
        if entry.get('eff_out_bw') is not None:
            swap_out_times[policies.index(policy)] = orig / (_safe_float(entry.get('eff_out_bw'), 0.0) * 1e9)
        if entry.get('eff_in_bw') is not None:
            swap_in_times[policies.index(policy)] = orig / (_safe_float(entry.get('eff_in_bw'), 0.0) * 1e9)

    return labels, swap_out_times, swap_in_times, ratios, max_errors


def make_summary_plot(summary, out_dir):
    model_labels = []
    all_swap_out = []
    all_swap_in = []
    all_ratios = []

    for model_id, data in summary.items():
        model_name = model_id.replace('/', '-')
        labels, swap_out_times, swap_in_times, ratios, _ = _build_policy_metrics(model_id, data)
        model_labels.append(model_name)
        all_swap_out.append(swap_out_times)
        all_swap_in.append(swap_in_times)
        all_ratios.append(ratios)

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'summary_cpu_gpu_transfer_comparison.png')

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CPU-GPU KV-cache transfer time under compression policies', fontsize=13, fontweight='bold')

    x = range(len(labels))
    width = 0.28

    for idx, model_name in enumerate(model_labels):
        offset = (idx - (len(model_labels) - 1) / 2) * width
        axes[0].bar([i + offset for i in x], all_swap_out[idx], width=width, label=model_name)
        axes[1].bar([i + offset for i in x], all_swap_in[idx], width=width, label=model_name)

    axes[0].set_title('Swap-out time: compressed vs. baseline')
    axes[0].set_ylabel('Transfer time (s)')
    axes[0].set_xticks(list(x))
    axes[0].set_xticklabels(labels, rotation=20, ha='right')
    axes[0].legend(ncol=2, fontsize=8)
    axes[0].set_ylim(bottom=0)

    axes[1].set_title('Swap-in time: compressed vs. baseline')
    axes[1].set_ylabel('Transfer time (s)')
    axes[1].set_xticks(list(x))
    axes[1].set_xticklabels(labels, rotation=20, ha='right')
    axes[1].legend(ncol=2, fontsize=8)
    axes[1].set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def plot_model(model_id, data, out_dir):
    model_id = model_id.replace('/', '-')
    os.makedirs(out_dir, exist_ok=True)
    orig = _safe_float(data.get('orig_size_bytes', 0), 0.0)

    policies = ['static_cuszp', 'adaptive_red_sim', 'int8', 'zlib']
    comp_sizes = [_safe_float(data[p].get('comp_size', 0), 0.0) for p in policies]
    eff_out = [_safe_float(data[p].get('eff_out_bw', 0.0), 0.0) for p in policies]

    ratios = [orig / s if s > 0 else 0 for s in comp_sizes]
    plt.figure(figsize=(6, 4))
    plt.bar(policies, ratios, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.ylabel('Compression ratio (orig / comp)')
    plt.title(f'{model_id}: Compression ratio')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_id}_compression_ratio.png'))
    plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(policies, eff_out, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
    plt.ylabel('Effective out bandwidth (GB/s)')
    plt.title(f'{model_id}: Effective out bandwidth')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'{model_id}_eff_out_bw.png'))
    plt.close()

    max_errors = [data[p].get('max_error', None) for p in policies]
    if any(x is not None for x in max_errors):
        plt.figure(figsize=(6, 4))
        xs = [x if x is not None else 0 for x in max_errors]
        plt.bar(policies, xs, color=['#4C72B0', '#55A868', '#C44E52', '#8172B2'])
        plt.ylabel('Max reconstruction error')
        plt.title(f'{model_id}: Max error')
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{model_id}_max_error.png'))
        plt.close()


def main():
    in_file = 'data/eval_summary.json'
    if len(sys.argv) > 1:
        in_file = sys.argv[1]

    with open(in_file, 'r') as fh:
        summary = json.load(fh)

    out_dir = os.path.join('data', 'figures')
    for model_id, data in summary.items():
        plot_model(model_id, data, out_dir)

    make_summary_plot(summary, out_dir)
    print(f'Wrote figures to {out_dir}')


if __name__ == '__main__':
    main()
