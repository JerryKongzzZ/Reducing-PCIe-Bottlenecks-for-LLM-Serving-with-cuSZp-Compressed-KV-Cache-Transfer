"""Build a jointly validated layer set for mixed-bound KV compression."""

import argparse
import copy
import json
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.layer_sensitivity_sweep import (
    MIN_CUSZP_ELEMENTS,
    compress_and_decompress_tensor,
    cuszp_wrapper_cpp,
)


def cache_layers(cache):
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return [(entry[0], entry[1]) for entry in list(cache)]


def rebuild_cache(original, replacements):
    from transformers.cache_utils import DynamicCache

    rebuilt = DynamicCache()
    if hasattr(original, "_seen_tokens"):
        rebuilt._seen_tokens = original._seen_tokens
    for layer_idx, (key, value) in enumerate(cache_layers(original)):
        selected_key, selected_value = replacements.get(layer_idx, (key, value))
        rebuilt.update(selected_key, selected_value, layer_idx=layer_idx)
    return rebuilt


def compress_selected_layers(compressor, original, selected, eps):
    layers = cache_layers(original)
    keys = torch.stack([layers[idx][0] for idx in selected], dim=1)
    values = torch.stack([layers[idx][1] for idx in selected], dim=1)
    # Preserve the runtime cross-layer order: [K/V, layer, ...].
    combined = torch.stack((keys, values), dim=0)
    restored, _, _, compressed_size, actual_eb = compress_and_decompress_tensor(
        compressor, combined, eps
    )
    replacements = {}
    for offset, layer_idx in enumerate(selected):
        replacements[layer_idx] = (
            restored[0, :, offset],
            restored[1, :, offset],
        )
    return replacements, compressed_size, actual_eb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--individual-profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--text-repeat", type=int, default=32)
    parser.add_argument("--probe-tokens", type=int, default=8)
    parser.add_argument("--error-bound", type=float, default=1e-2)
    parser.add_argument("--kl-threshold", type=float, default=1e-2)
    parser.add_argument("--min-top1-match", type=float, default=1.0)
    args = parser.parse_args()

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model).to(device).eval()
    tokenized = tokenizer(
        args.text * args.text_repeat,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )
    input_ids = tokenized["input_ids"].to(device)
    input_attention_mask = tokenized["attention_mask"].to(device)
    with torch.no_grad():
        generated = model.generate(
            input_ids,
            attention_mask=input_attention_mask,
            max_new_tokens=max(0, args.probe_tokens - 1),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
        original = model(
            input_ids[:, :-1],
            attention_mask=input_attention_mask[:, :-1],
            use_cache=True,
        ).past_key_values
    continuation = generated[:, input_ids.size(1):]
    target_tokens = torch.cat((input_ids[:, -1:], continuation), dim=1)
    probe_attention_mask = torch.ones(
        (target_tokens.size(0), input_ids.size(1) - 1 + target_tokens.size(1)),
        dtype=input_attention_mask.dtype,
        device=device,
    )
    with torch.no_grad():
        baseline = model(
            input_ids=target_tokens,
            attention_mask=probe_attention_mask,
            past_key_values=copy.deepcopy(original),
        ).logits.float()

    config = cuszp_wrapper_cpp.CompressionConfig(
        error_bound=args.error_bound,
        use_relative_error=True,
        processing_dim=cuszp_wrapper_cpp.CuszpDim.DIM_1D,
        encoding_mode=cuszp_wrapper_cpp.CuszpMode.MODE_PLAIN,
        data_type=cuszp_wrapper_cpp.CuszpType.TYPE_FLOAT,
    )
    compressor = cuszp_wrapper_cpp.CuSZpWrapper(config, 0)
    individual = json.loads(Path(args.individual_profile).read_text())["layers"]
    detail_at_bound = {}
    for layer_idx, entry in individual.items():
        matching = [
            item for item in entry["details"]
            if abs(float(item["eps"]) - args.error_bound) < 1e-12
        ]
        detail_at_bound[int(layer_idx)] = (
            matching[0]["kl"] if matching else float("inf")
        )
    remaining = sorted(detail_at_bound, key=detail_at_bound.get)
    selected = []
    trace = []

    while remaining:
        best = None
        for candidate in remaining:
            trial = selected + [candidate]
            replacements, compressed_size, actual_eb = compress_selected_layers(
                compressor, original, trial, args.error_bound
            )
            with torch.no_grad():
                logits = model(
                    input_ids=target_tokens,
                    attention_mask=probe_attention_mask,
                    past_key_values=rebuild_cache(original, replacements),
                ).logits.float()
            kl = float(
                F.kl_div(
                    F.log_softmax(logits, dim=-1),
                    F.softmax(baseline, dim=-1),
                    reduction="none",
                ).sum(dim=-1).mean().item()
            )
            top1 = float(
                (logits.argmax(dim=-1) == baseline.argmax(dim=-1))
                .float().mean().item()
            )
            safe = kl <= args.kl_threshold and top1 >= args.min_top1_match
            record = {
                "candidate": candidate,
                "selected_layers": trial,
                "kl": kl,
                "top1_match": top1,
                "compressed_bytes": int(compressed_size),
                "actual_error_bound": float(actual_eb),
                "safe": safe,
            }
            if safe and (best is None or kl < best[0]):
                best = (kl, candidate, record)
        if best is None:
            break
        _, chosen, record = best
        selected.append(chosen)
        remaining.remove(chosen)
        trace.append(record)
        print(
            f"selected layer {chosen}: count={len(selected)} "
            f"kl={record['kl']:.4e} top1={record['top1_match']:.3f}"
        )

    layers = {}
    for layer_idx in range(len(cache_layers(original))):
        enabled = layer_idx in selected
        layers[str(layer_idx)] = {
            "category": "deep" if enabled else "shallow",
            "max_safe_eps": args.error_bound if enabled else 0.0,
            "individual": individual[str(layer_idx)],
        }
    result = {
        "_metadata": {
            "model": args.model,
            "method": "joint_greedy_continuation_validation",
            "error_bound": args.error_bound,
            "kl_threshold": args.kl_threshold,
            "min_top1_match": args.min_top1_match,
            "probe_tokens": args.probe_tokens,
            "minimum_runtime_layers": (MIN_CUSZP_ELEMENTS + 4095) // 4096,
            "selected_layers": selected,
            "runtime_eligible": len(selected) >= (MIN_CUSZP_ELEMENTS + 4095) // 4096,
            "trace": trace,
        },
        "layers": layers,
    }
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result["_metadata"], indent=2))


if __name__ == "__main__":
    main()
