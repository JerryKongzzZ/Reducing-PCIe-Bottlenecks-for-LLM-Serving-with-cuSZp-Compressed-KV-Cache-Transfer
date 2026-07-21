"""Build a worst-case layer sensitivity profile from multiple prompts.

Each prompt is calibrated independently by ``layer_sensitivity_sweep.py`` so
the raw evidence remains inspectable. This script then accepts an error bound
only when every prompt satisfies both the KL and top-1 constraints. It is a
calibration step, not a replacement for the full vLLM workload quality gate.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
SWEEP = REPO_ROOT / "benchmarks" / "layer_sensitivity_sweep.py"
sys.path.insert(0, str(REPO_ROOT))

from benchmarks.layer_sensitivity_sweep import assign_sensitivity_categories


def load_prompt_suite(path):
    """Load and validate a JSON prompt list."""
    prompt_path = Path(path)
    payload = json.loads(prompt_path.read_text(encoding="utf-8"))
    entries = payload.get("prompts") if isinstance(payload, dict) else payload
    if not isinstance(entries, list) or len(entries) < 2:
        raise ValueError("prompt suite must contain at least two prompts")
    prompts = []
    seen = set()
    for index, entry in enumerate(entries):
        if isinstance(entry, str):
            prompt_id, text, repeat = f"prompt_{index}", entry, 1
        elif isinstance(entry, dict):
            prompt_id = str(entry.get("id", f"prompt_{index}"))
            text = entry.get("text")
            repeat = entry.get("repeat", 1)
        else:
            raise ValueError(f"prompt {index} must be a string or object")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"prompt {prompt_id!r} has no text")
        if not isinstance(repeat, int) or isinstance(repeat, bool) or repeat < 1:
            raise ValueError(
                f"prompt {prompt_id!r} repeat must be a positive integer"
            )
        if prompt_id in seen:
            raise ValueError(f"duplicate prompt id: {prompt_id}")
        seen.add(prompt_id)
        prompts.append({"id": prompt_id, "text": text, "repeat": repeat})
    return prompts


def prompt_suite_fingerprint(prompts):
    canonical = json.dumps(
        prompts, ensure_ascii=False, sort_keys=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reproducible_path(path):
    """Record repository inputs without machine-specific absolute prefixes."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(resolved)


def _detail_by_bound(layer_entry):
    return {float(item["eps"]): item for item in layer_entry["details"]}


def merge_profiles(
    named_profiles, *, kl_threshold, min_top1_match, error_bounds=None
):
    """Merge per-prompt profiles using worst-case quality measurements."""
    if len(named_profiles) < 2:
        raise ValueError("at least two per-prompt profiles are required")
    prompt_ids = [prompt_id for prompt_id, _ in named_profiles]
    if len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("per-prompt profile ids must be unique")
    first = named_profiles[0][1]
    layer_ids = set(first.get("layers", {}))
    if not layer_ids:
        raise ValueError("profile has no layers")
    model = first.get("_metadata", {}).get("model")
    for prompt_id, profile in named_profiles[1:]:
        if set(profile.get("layers", {})) != layer_ids:
            raise ValueError(f"layer set mismatch in prompt {prompt_id}")
        other_model = profile.get("_metadata", {}).get("model")
        if model and other_model and other_model != model:
            raise ValueError(f"model mismatch in prompt {prompt_id}")

    merged_layers = {}
    for layer_id in sorted(layer_ids, key=int):
        bound_sets = [
            set(_detail_by_bound(profile["layers"][layer_id]))
            for _, profile in named_profiles
        ]
        if any(bounds != bound_sets[0] for bounds in bound_sets[1:]):
            raise ValueError(f"error-bound set mismatch for layer {layer_id}")
        selected_bounds = (
            sorted(bound_sets[0])
            if error_bounds is None
            else sorted(set(float(value) for value in error_bounds))
        )
        missing = set(selected_bounds) - bound_sets[0]
        if missing:
            raise ValueError(
                f"missing requested error bounds for layer {layer_id}: "
                f"{sorted(missing)}"
            )
        details = []
        for eps in selected_bounds:
            observations = []
            for prompt_id, profile in named_profiles:
                item = _detail_by_bound(profile["layers"][layer_id])[eps]
                observations.append({
                    "prompt_id": prompt_id,
                    "kl": float(item["kl"]),
                    "top1_match": float(item["top1_match"]),
                    "k_size": int(item["k_size"]),
                    "v_size": int(item["v_size"]),
                    "k_actual_eb": float(item["k_actual_eb"]),
                    "v_actual_eb": float(item["v_actual_eb"]),
                })
            worst_kl = max(item["kl"] for item in observations)
            min_top1 = min(item["top1_match"] for item in observations)
            details.append({
                "eps": eps,
                "kl": worst_kl,
                "mean_kl": (
                    sum(item["kl"] for item in observations) / len(observations)
                ),
                "top1_match": min_top1,
                "safe": (
                    worst_kl <= kl_threshold
                    and min_top1 >= min_top1_match
                ),
                "prompt_metrics": observations,
            })
        max_safe_eps = 0.0
        for detail in details:
            if not detail["safe"]:
                break
            max_safe_eps = float(detail["eps"])
        merged_layers[layer_id] = {
            "score": max(item["kl"] for item in details),
            "max_safe_eps": max_safe_eps,
            "category": None,
            "details": details,
        }
    assign_sensitivity_categories(merged_layers)
    return {
        "_metadata": {
            "model": model,
            "method": "worst_case_multi_prompt_teacher_forced_calibration",
            "prompt_count": len(named_profiles),
            "prompt_ids": prompt_ids,
            "kl_threshold": kl_threshold,
            "min_top1_match": min_top1_match,
            "error_bounds": [
                float(item["eps"])
                for item in next(iter(merged_layers.values()))["details"]
            ],
            "quality_aggregation": {
                "kl": "maximum_across_prompts",
                "top1_match": "minimum_across_prompts",
                "safe_prefix_required": True,
            },
        },
        "layers": merged_layers,
    }


def _safe_stem(prompt_id):
    return (
        re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_id).strip("_")
        or "prompt"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--prompts-file", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile-dir")
    parser.add_argument(
        "--eps", type=float, nargs="+", default=(1e-5, 1e-4)
    )
    parser.add_argument("--probe-tokens", type=int, default=8)
    parser.add_argument("--kl-threshold", type=float, default=1e-2)
    parser.add_argument("--min-top1-match", type=float, default=1.0)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--keep-existing", action="store_true")
    args = parser.parse_args()

    prompts = load_prompt_suite(args.prompts_file)
    out_path = Path(args.out).resolve()
    profile_dir = (
        Path(args.profile_dir).resolve()
        if args.profile_dir
        else out_path.parent / f"{out_path.stem}_per_prompt"
    )
    profile_dir.mkdir(parents=True, exist_ok=True)
    named_profiles = []
    source_profiles = []
    for index, prompt in enumerate(prompts):
        profile_path = profile_dir / (
            f"{index:02d}_{_safe_stem(prompt['id'])}.json"
        )
        if not args.aggregate_only and not (
            args.keep_existing and profile_path.exists()
        ):
            command = [
                sys.executable,
                str(SWEEP),
                "--model", args.model,
                "--text", prompt["text"],
                "--text-repeat", str(prompt["repeat"]),
                "--probe-tokens", str(args.probe_tokens),
                "--out", str(profile_path),
                "--eps", *[
                    str(value) for value in sorted(set(args.eps))
                ],
                "--device", str(args.device),
                "--kl-threshold", str(args.kl_threshold),
                "--min-top1-match", str(args.min_top1_match),
            ]
            print(
                f"calibrating prompt {index + 1}/{len(prompts)}: "
                f"{prompt['id']}",
                flush=True,
            )
            subprocess.run(command, cwd=REPO_ROOT, check=True)
        if not profile_path.exists():
            parser.error(f"missing per-prompt profile: {profile_path}")
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        named_profiles.append((prompt["id"], profile))
        source_profiles.append(reproducible_path(profile_path))

    result = merge_profiles(
        named_profiles,
        kl_threshold=args.kl_threshold,
        min_top1_match=args.min_top1_match,
        error_bounds=args.eps,
    )
    result["_metadata"].update({
        "prompt_suite": reproducible_path(args.prompts_file),
        "prompt_suite_sha256": prompt_suite_fingerprint(prompts),
        "probe_tokens": args.probe_tokens,
        "source_profiles": source_profiles,
        "full_workload_validation_required": True,
    })
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["_metadata"], indent=2), flush=True)


if __name__ == "__main__":
    main()
