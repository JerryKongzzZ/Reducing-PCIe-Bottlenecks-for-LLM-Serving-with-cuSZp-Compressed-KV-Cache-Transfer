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
import copy
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


MIN_CUSZP_ELEMENTS = 36864


def assign_sensitivity_categories(results):
    """Assign three relative sensitivity bins from measured KL scores.

    The legacy names are kept for scheduler compatibility: ``shallow`` means
    most sensitive (smallest error bound), while ``deep`` means least
    sensitive (largest permissible error bound).  Relative ranking is more
    robust than the previous hard-coded KL thresholds, which classified every
    GPT-2 layer into the same bin.
    """
    if results and all('max_safe_eps' in entry for entry in results.values()):
        # A smaller safe bound means greater sensitivity.  Break ties with the
        # observed worst-case KL so the bins remain deterministic.
        ranked = sorted(
            results,
            key=lambda idx: (
                results[idx]['max_safe_eps'],
                -results[idx]['score'],
            ),
        )
    else:
        ranked = sorted(results, key=lambda idx: results[idx]['score'], reverse=True)
    n_layers = len(ranked)
    n_high = max(1, (n_layers + 3) // 4)
    n_mid = max(1, (n_layers + 3) // 4) if n_layers > 1 else 0
    for rank, layer_idx in enumerate(ranked):
        if rank < n_high:
            category = 'shallow'
        elif rank < n_high + n_mid:
            category = 'mid'
        else:
            category = 'deep'
        results[layer_idx]['category'] = category


def compress_and_decompress_tensor(compressor, tensor: torch.Tensor, eps: float):
    device = tensor.device
    original_shape = tuple(tensor.shape)
    original_dtype = tensor.dtype
    source = tensor.to(torch.float32).contiguous().view(-1)
    original_numel = source.numel()
    if original_numel < MIN_CUSZP_ELEMENTS:
        source = F.pad(source, (0, MIN_CUSZP_ELEMENTS - original_numel))
    estimated_size = cuszp_wrapper_cpp.CuSZpWrapper.estimate_compressed_buffer_size(source.numel() * source.element_size())
    compressed_buffer = torch.empty(estimated_size, dtype=torch.uint8, device=device)
    # compress
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    success, compressed_buffer, compressed_size, actual_eb = compressor.compress(source, compressed_buffer, float(eps))
    torch.cuda.synchronize()
    comp_time = time.perf_counter() - t0

    if not success:
        raise RuntimeError('compression error')

    # decompress
    decomp = torch.empty_like(source)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ok = compressor.decompress(compressed_buffer, int(compressed_size), decomp, float(actual_eb))
    torch.cuda.synchronize()
    decomp_time = time.perf_counter() - t0
    if not ok:
        raise RuntimeError('decompression error')

    restored = decomp[:original_numel].view(original_shape).to(original_dtype)
    return restored, comp_time, decomp_time, int(compressed_size), float(actual_eb)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--text', type=str, default=None)
    parser.add_argument('--text-repeat', type=int, default=8)
    parser.add_argument(
        '--probe-tokens',
        type=int,
        default=8,
        help='Number of final tokens used for teacher-forced KL calibration.',
    )
    parser.add_argument('--out', type=str, default='data/layer_sensitivity.json')
    parser.add_argument('--eps', type=float, nargs='+', default=[1e-5, 1e-4, 1e-3, 1e-2])
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument(
        '--kl-threshold',
        type=float,
        default=1e-2,
        help='Maximum next-token KL allowed when deriving each layer safe bound.',
    )
    parser.add_argument(
        '--min-top1-match',
        type=float,
        default=1.0,
        help='Minimum agreement with baseline greedy tokens for a safe bound.',
    )
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.device}')

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device)
    model.eval()

    text_seed = args.text or "The Hong Kong Polytechnic University (PolyU) is a public research university. "
    text = text_seed * args.text_repeat
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    input_ids = inputs['input_ids']
    input_attention_mask = inputs['attention_mask'].to(device)

    if args.probe_tokens < 1 or input_ids.size(1) < 2:
        raise RuntimeError('Need at least two input tokens and one probe token')

    # Match the actual greedy decode path.  Cache the full prompt except its
    # final token, then teacher-force the final prompt token followed by the
    # baseline continuation.  The resulting logits predict the same tokens a
    # vLLM replay will greedily generate.
    if args.probe_tokens > 1:
        with torch.no_grad():
            generated = model.generate(
                input_ids.to(device),
                attention_mask=input_attention_mask,
                max_new_tokens=args.probe_tokens - 1,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        continuation = generated[:, input_ids.size(1):]
    else:
        continuation = input_ids[:, :0].to(device)
    context_ids = input_ids[:, :-1].to(device)
    target_tokens = torch.cat(
        (input_ids[:, -1:].to(device), continuation), dim=1
    )

    # Warm-up and compute original past_key_values
    with torch.no_grad():
        out = model(
            context_ids,
            attention_mask=input_attention_mask[:, :-1],
            use_cache=True,
        )
    orig_past = out.past_key_values

    # Baseline logits for next token. DynamicCache is mutated in place during
    # a forward pass, so never pass orig_past directly here; doing so polluted
    # every subsequent sensitivity measurement.
    baseline_past = copy.deepcopy(orig_past)
    with torch.no_grad():
        probe_attention_mask = torch.ones(
            (target_tokens.size(0), context_ids.size(1) + target_tokens.size(1)),
            dtype=input_attention_mask.dtype,
            device=device,
        )
        baseline = model(
            input_ids=target_tokens,
            attention_mask=probe_attention_mask,
            past_key_values=baseline_past,
        )
    baseline_logits = baseline.logits

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
    if type(orig_past).__name__ == "DynamicCache" and not hasattr(orig_past, "key_cache"):
        num_layers = len(list(orig_past))
    results = {}

    for layer_idx in range(num_layers):
        print(f"Profiling layer {layer_idx}/{num_layers - 1}")
        scores = []
        for eps in args.eps:
            if type(orig_past).__name__ == "DynamicCache" or hasattr(orig_past, "key_cache"):
                if hasattr(orig_past, "key_cache"):
                    k_tensor = orig_past.key_cache[layer_idx].detach().clone().to(device)
                    v_tensor = orig_past.value_cache[layer_idx].detach().clone().to(device)
                else:
                    k_tensor = list(orig_past)[layer_idx][0].detach().clone().to(device)
                    v_tensor = list(orig_past)[layer_idx][1].detach().clone().to(device)
            else:
                k_tensor = orig_past[layer_idx][0].detach().clone().to(device)
                v_tensor = orig_past[layer_idx][1].detach().clone().to(device)

            # compress+decompress keys and values separately
            k_decomp, kt_comp, kt_decomp_time, k_size, k_actual_eb = compress_and_decompress_tensor(compressor, k_tensor, eps)
            v_decomp, vt_comp, vt_decomp_time, v_size, v_actual_eb = compress_and_decompress_tensor(compressor, v_tensor, eps)

            # place back
            if type(orig_past).__name__ == "DynamicCache" or hasattr(orig_past, "key_cache"):
                from transformers.cache_utils import DynamicCache
                past_tuple = DynamicCache()
                if hasattr(orig_past, "_seen_tokens"):
                    past_tuple._seen_tokens = orig_past._seen_tokens
                for i in range(num_layers):
                    if i == layer_idx:
                        past_tuple.update(k_decomp, v_decomp, layer_idx=i)
                    else:
                        if hasattr(orig_past, "key_cache"):
                            past_tuple.update(orig_past.key_cache[i], orig_past.value_cache[i], layer_idx=i)
                        else:
                            past_tuple.update(list(orig_past)[i][0], list(orig_past)[i][1], layer_idx=i)
            else:
                past_copy = list(list(x) for x in orig_past)
                past_copy[layer_idx][0] = k_decomp
                past_copy[layer_idx][1] = v_decomp
                past_tuple = tuple(tuple(x) for x in past_copy)

            # Run model forward for the target token with modified past
            with torch.no_grad():
                modified = model(
                    input_ids=target_tokens,
                    attention_mask=probe_attention_mask,
                    past_key_values=past_tuple,
                )
            mod_logits = modified.logits

            # Compute KL divergence between original distribution and modified distribution
            p = F.softmax(baseline_logits.float(), dim=-1)
            q_log = F.log_softmax(mod_logits.float(), dim=-1)
            kl = F.kl_div(q_log, p, reduction='none').sum(dim=-1).mean().item()
            top1_match = float(
                (mod_logits.argmax(dim=-1) == baseline_logits.argmax(dim=-1))
                .float()
                .mean()
                .item()
            )
            if not torch.isfinite(torch.tensor(kl)):
                raise RuntimeError(
                    f"non-finite KL at layer={layer_idx}, eps={eps}; "
                    "the sensitivity result is invalid"
                )

            scores.append({'eps': eps, 'kl': kl, 'top1_match': top1_match, 'k_size': k_size, 'v_size': v_size, 'k_actual_eb': k_actual_eb, 'v_actual_eb': v_actual_eb})
            print(f"  eps={eps:.0e} kl={kl:.4e} top1={top1_match:.3f} k_size={k_size} v_size={v_size}")

        # The safe bound is the largest prefix value for which this bound and
        # every tighter tested bound satisfy the quality constraint.  Requiring
        # a safe prefix avoids selecting a looser point because of measurement
        # noise when an intermediate point already violated the constraint.
        max_kl = max(s['kl'] for s in scores)
        max_safe_eps = 0.0
        for score in sorted(scores, key=lambda item: item['eps']):
            if (
                score['kl'] > args.kl_threshold
                or score['top1_match'] < args.min_top1_match
            ):
                break
            max_safe_eps = float(score['eps'])
        results[layer_idx] = {
            'score': max_kl,
            'max_safe_eps': max_safe_eps,
            'category': None,
            'details': scores,
        }

    assign_sensitivity_categories(results)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    with open(args.out, 'w') as fh:
        json.dump(
            {
                '_metadata': {
                    'model': args.model,
                    'kl_threshold': args.kl_threshold,
                    'error_bounds': sorted(args.eps),
                    'metric': 'mean_teacher_forced_token_kl',
                    'probe_tokens': args.probe_tokens,
                    'min_top1_match': args.min_top1_match,
                    'calibration_text_repeat': args.text_repeat,
                    'category_semantics': {
                        'shallow': 'most_sensitive',
                        'mid': 'medium_sensitivity',
                        'deep': 'least_sensitive',
                    },
                },
                'layers': results,
            },
            fh,
            indent=2,
        )

    print(f"Wrote layer sensitivity mapping to {args.out}")


if __name__ == '__main__':
    main()
