"""
Produce a per-layer sensitivity mapping (layer_sensitivity.json).

For each layer, we measure impact of compressing that layer's KV tensors
on the next-token distribution (KL divergence). The script outputs a
JSON mapping layer_idx -> {"score": float, "category": str} where
category in {"shallow","mid","deep"}.

Usage:
  PYTHONPATH=integration/compression_pipeline python3 benchmarks/layer_sensitivity_sweep.py \
    --model gpt2 --out data/layer_sensitivity.json --eps 1e-5 1e-4 1e-3 1e-2

Notes:
  - Requires `cuszp_wrapper_cpp` compiled and importable.
  - Results are heuristic-driven; tune thresholds for your models.
"""
import argparse
import json
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
pipeline_dir = os.path.abspath(os.path.join(current_dir, '..', 'integration', 'compression_pipeline'))
if pipeline_dir not in sys.path:
    sys.path.insert(0, pipeline_dir)

try:
    import cuszp_wrapper_cpp
except Exception as e:
    raise RuntimeError(f"cuszp_wrapper_cpp import failed: {e}")


def classify_score(score: float, t_shallow=1.0, t_mid=0.1) -> str:
    if score >= t_shallow:
        return 'shallow'
    if score >= t_mid:
        return 'mid'
    return 'deep'


def compress_and_decompress_tensor(compressor, tensor: torch.Tensor, eps: float):
    device = tensor.device
    estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(tensor.numel() * tensor.element_size())
    compressed_buffer = torch.empty(estimated_size, dtype=torch.uint8, device=device)
    # compress
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    success, compressed_buffer, compressed_size, actual_eb = compressor.compress(tensor, compressed_buffer, float(eps))
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - t0

    if not success:
        raise RuntimeError('compression error')

    # decompress
    decomp = torch.empty_like(tensor)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = compressor.decompress(compressed_buffer, int(compressed_size), decomp, float(actual_eb))
    torch.cuda.synchronize()
    decomp_time = time.perf_counter() - t0
    if not ok:
        raise RuntimeError('decompression error')

    return decomp, comp_time, decomp_time, int(compressed_size), float(actual_eb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--out', type=str, default='data/layer_sensitivity.json')
    parser.add_argument('--eps', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument('--device', type=int, default=0)
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    text = args.text or ("The Hong Kong Polytechnic University (PolyU) is a public research university. " * 8)
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids']

    # Build context (all but last token) and target (last token)
    if input_ids.size(1) < 2:
        raise RuntimeError('Need at least 2 tokens to form context+target')

    context_ids = input_ids[:, :-1].to(device)
    target_token = input_ids[:, -1:].to(device)

    # Warm-up and compute original past_key_values
    with torch.no_grad():
        out = model(context_ids, use_cache=True)
    orig_past = out.past_key_values

    # Baseline logits for next token
    with torch.no_grad():
        baseline = model(input_ids=target_token, past_key_values=orig_past)
    baseline_logits = baseline.logits[:, 0, :]

    # Initialize compressor
    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=1e-4,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, args.device)

    num_layers = len(orig_past) if not hasattr(orig_past, "key_cache") else len(orig_past.key_cache)
    results = {}

    for layer_idx in range(num_layers):
        print(f"Profiling layer {layer_idx}/{num_layers - 1}")
        scores = []
        for eps in args.eps:
            if hasattr(orig_past, "key_cache"):
                k_tensor = orig_past.key_cache[layer_idx].detach().clone().to(device)
                v_tensor = orig_past.value_cache[layer_idx].detach().clone().to(device)
            else:
                k_tensor = orig_past[layer_idx][0].detach().clone().to(device)
                v_tensor = orig_past[layer_idx][1].detach().clone().to(device)

            # compress+decompress keys and values separately
            k_decomp, kt_comp, kt_decomp_time, k_size, k_actual_eb = compress_and_decompress_tensor(compressor, k_tensor, eps)
            v_decomp, vt_comp, vt_decomp_time, v_size, v_actual_eb = compress_and_decompress_tensor(compressor, v_tensor, eps)

            # place back
            if hasattr(orig_past, "key_cache"):
                from transformers.cache_utils import DynamicCache
                past_tuple = DynamicCache()
                if hasattr(orig_past, "_seen_tokens"):
                    past_tuple._seen_tokens = orig_past._seen_tokens
                for i in range(num_layers):
                    if i == layer_idx:
                        past_tuple.key_cache.append(k_decomp)
                        past_tuple.value_cache.append(v_decomp)
                    else:
                        past_tuple.key_cache.append(orig_past.key_cache[i])
                        past_tuple.value_cache.append(orig_past.value_cache[i])
            else:
                past_copy = list(list(x) for x in orig_past)
                past_copy[layer_idx][0] = k_decomp
                past_copy[layer_idx][1] = v_decomp
                past_tuple = tuple(tuple(x) for x in past_copy)

            # Run model forward for the target token with modified past
            with torch.no_grad():
                modified = model(input_ids=target_token, past_key_values=past_tuple)
            mod_logits = modified.logits[:, 0, :]

            # Compute KL divergence between original distribution and modified distribution
            p = F.softmax(baseline_logits, dim=-1)
            q_log = F.log_softmax(mod_logits, dim=-1)
            kl = F.kl_div(q_log, p, reduction='batchmean').item()

            scores.append({'eps': eps, 'kl': kl, 'k_size': k_size, 'v_size': v_size, 'k_actual_eb': k_actual_eb, 'v_actual_eb': v_actual_eb})
            print(f"  eps={eps:.0e} kl={kl:.4e} k_size={k_size} v_size={v_size}")

        # Summarize sensitivity: take maximum KL across eps levels
        max_kl = max(s['kl'] for s in scores)
        category = classify_score(max_kl)
        results[layer_idx] = {'score': max_kl, 'category': category, 'details': scores}

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(results, fh, indent=2)

    print(f"Wrote layer sensitivity mapping to {args.out}")


if __name__ == '__main__':
    main()
