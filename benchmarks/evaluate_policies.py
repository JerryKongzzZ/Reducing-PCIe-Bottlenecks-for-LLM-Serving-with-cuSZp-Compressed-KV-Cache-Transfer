"""
Evaluate compression policies (static cuSZp, adaptive cuSZp, INT8, zlib) on KV caches.

Usage:
  PYTHONPATH=integration/compression_pipeline python3 benchmarks/evaluate_policies.py --models gpt2 --out data/eval_summary.json

This script relies on `benchmarks/benchmark_pipeline.py` helper functions and the
`cuszp_wrapper_cpp` extension (rebuild required after C++ changes).
"""
import argparse
import json
import os
import sys
import time
import zlib

import numpy as np
import torch

current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import benchmark_pipeline as bp
from benchmark_pipeline import generate_kv_cache, measure_baseline, measure_compression

try:
    import cuszp_wrapper_cpp
except Exception as e:
    raise RuntimeError(f"cuszp_wrapper_cpp import failed: {e}")


def int8_quantize_dequantize(tensor: torch.Tensor):
    t0 = time.perf_counter()
    cpu = tensor.cpu()
    amin = float(cpu.min())
    amax = float(cpu.max())
    scale = max(abs(amin), abs(amax)) / 127.0 if max(abs(amin), abs(amax)) > 0 else 1.0
    q = (cpu / scale).round().clamp(-128, 127).to(torch.int8)
    comp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    deq = q.to(torch.float32) * scale
    decomp_time = time.perf_counter() - t0
    comp_size = q.numel()  # bytes per int8
    return deq.to(tensor.device), comp_time, decomp_time, comp_size


def zlib_compress_decompress(tensor: torch.Tensor):
    t0 = time.perf_counter()
    b = tensor.cpu().numpy().tobytes()
    c = zlib.compress(b)
    comp_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    _ = zlib.decompress(c)
    decomp_time = time.perf_counter() - t0
    comp_size = len(c)
    return comp_time, decomp_time, comp_size


def simulate_adaptive_on_flat_tensor(tensor: torch.Tensor, compressor, scheduler_policy, num_slices=8, device_id=0):
    # Partition tensor into slices and compress each slice with assigned eps
    n = tensor.numel()
    slice_len = n // num_slices
    comp_total = 0
    comp_time_total = 0.0
    decomp_time_total = 0.0

    for i in range(num_slices):
        start = i * slice_len
        end = n if i == num_slices - 1 else (i + 1) * slice_len
        view = tensor.view(-1)[start:end].contiguous().to(torch.device(f"cuda:{device_id}"))
        # Map slice index to sensitivity category via round-robin of policy keys if necessary
        categories = list(scheduler_policy.keys())
        idx = i % len(categories)
        # scheduler_policy is an ordered list of categories (e.g., ['shallow','mid','deep']) -> pick eps per category
        cat = categories[idx]
        eps = scheduler_policy[cat]
        estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(view.numel() * view.element_size())
        compressed_buffer = torch.empty(estimated_size, dtype=torch.uint8, device=view.device)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        success, compressed_buffer, compressed_size, actual_eb = compressor.compress(view, compressed_buffer, float(eps))
        torch.cuda.synchronize()
        comp_t = time.perf_counter() - t0

        if not success:
            raise RuntimeError('compress failed in simulate_adaptive_on_flat_tensor')

        # decompress to measure cost
        decomp = torch.empty_like(view)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        ok = compressor.decompress(compressed_buffer, int(compressed_size), decomp, float(actual_eb))
        torch.cuda.synchronize()
        decomp_t = time.perf_counter() - t0
        if not ok:
            raise RuntimeError('decompress failed in simulate_adaptive_on_flat_tensor')

        comp_total += int(compressed_size)
        comp_time_total += comp_t
        decomp_time_total += decomp_t

    return comp_total, comp_time_total, decomp_time_total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', nargs='+', default=['gpt2'])
    parser.add_argument('--out', type=str, default='data/eval_summary.json')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--iterations', type=int, default=20)
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.device}")
    torch.cuda.set_device(device)

    # Prepare cuSZp compressor with default config; we will use per-call eps override where needed
    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=1e-4,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, args.device)

    # Policies to evaluate
    policies = ['baseline', 'static_cuszp', 'adaptive_cuszp', 'int8', 'zlib']

    # Adaptive policy mapping for a given congestion state (we evaluate in RED mode by default)
    adaptive_red = {'shallow': 1e-4, 'mid': 1e-3, 'deep': 1e-2}
    adaptive_yellow = {'shallow': 1e-5, 'mid': 1e-5, 'deep': 1e-4}
    adaptive_green = {'shallow': 1e-5, 'mid': 1e-5, 'deep': 1e-5}

    results = {}

    for model_id in args.models:
        print(f"\nEvaluating model {model_id}")
        tensor = generate_kv_cache(model_id, target_size=4194304).to(device)
        orig_size_bytes = tensor.numel() * tensor.element_size()

        # baseline bandwidths
        base_perf = measure_baseline(tensor, device, num_iterations=args.iterations)

        # static cuSZp (eps=1e-4)
        comp_perf = measure_compression(tensor, args.device, error_bound=1e-4, num_iterations=args.iterations)
        # adaptive red simulation (simulate slices)
        comp_total_red, comp_time_red, decomp_time_red = simulate_adaptive_on_flat_tensor(tensor, compressor, adaptive_red, num_slices=8, device_id=args.device)

        # INT8
        deq, q_comp_t, q_decomp_t, q_size = int8_quantize_dequantize(tensor)
        max_err_q = float(torch.max(torch.abs(tensor - deq)).item())

        # zlib
        z_comp_t, z_decomp_t, z_size = zlib_compress_decompress(tensor)

        # Summaries
        res = {
            'orig_size_bytes': int(orig_size_bytes),
            'baseline': base_perf,
            'static_cuszp': {
                'comp_time': comp_perf['comp_time'],
                'decomp_time': comp_perf['decomp_time'],
                'comp_size': comp_perf['comp_size'],
                'max_error': comp_perf['max_error']
            },
            'adaptive_red_sim': {
                'comp_size': int(comp_total_red),
                'comp_time': float(comp_time_red),
                'decomp_time': float(decomp_time_red)
            },
            'int8': {
                'comp_time': float(q_comp_t),
                'decomp_time': float(q_decomp_t),
                'comp_size': int(q_size),
                'max_error': float(max_err_q)
            },
            'zlib': {
                'comp_time': float(z_comp_t),
                'decomp_time': float(z_decomp_t),
                'comp_size': int(z_size)
            }
        }

        # Effective bandwidth calculations (similar to benchmark_pipeline)
        # static cuSZp
        transfer_out_time = res['static_cuszp']['comp_size'] / (base_perf['d2h_bandwidth'] * 1e9)
        eff_out_time = res['static_cuszp']['comp_time'] + transfer_out_time
        res['static_cuszp']['eff_out_bw'] = (orig_size_bytes / eff_out_time) / 1e9

        transfer_in_time = res['static_cuszp']['comp_size'] / (base_perf['h2d_bandwidth'] * 1e9)
        eff_in_time = transfer_in_time + res['static_cuszp']['decomp_time']
        res['static_cuszp']['eff_in_bw'] = (orig_size_bytes / eff_in_time) / 1e9

        # adaptive red
        transfer_out_time = res['adaptive_red_sim']['comp_size'] / (base_perf['d2h_bandwidth'] * 1e9)
        eff_out_time = res['adaptive_red_sim']['comp_time'] + transfer_out_time
        res['adaptive_red_sim']['eff_out_bw'] = (orig_size_bytes / eff_out_time) / 1e9

        transfer_in_time = res['adaptive_red_sim']['comp_size'] / (base_perf['h2d_bandwidth'] * 1e9)
        eff_in_time = transfer_in_time + res['adaptive_red_sim']['decomp_time']
        res['adaptive_red_sim']['eff_in_bw'] = (orig_size_bytes / eff_in_time) / 1e9

        # int8
        transfer_out_time = res['int8']['comp_size'] / (base_perf['d2h_bandwidth'] * 1e9)
        eff_out_time = res['int8']['comp_time'] + transfer_out_time
        res['int8']['eff_out_bw'] = (orig_size_bytes / eff_out_time) / 1e9

        transfer_in_time = res['int8']['comp_size'] / (base_perf['h2d_bandwidth'] * 1e9)
        eff_in_time = transfer_in_time + res['int8']['decomp_time']
        res['int8']['eff_in_bw'] = (orig_size_bytes / eff_in_time) / 1e9

        # zlib
        transfer_out_time = res['zlib']['comp_size'] / (base_perf['d2h_bandwidth'] * 1e9)
        eff_out_time = res['zlib']['comp_time'] + transfer_out_time
        res['zlib']['eff_out_bw'] = (orig_size_bytes / eff_out_time) / 1e9

        transfer_in_time = res['zlib']['comp_size'] / (base_perf['h2d_bandwidth'] * 1e9)
        eff_in_time = transfer_in_time + res['zlib']['decomp_time']
        res['zlib']['eff_in_bw'] = (orig_size_bytes / eff_in_time) / 1e9

        results[model_id] = res

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote evaluation summary to {args.out}")


if __name__ == '__main__':
    main()
