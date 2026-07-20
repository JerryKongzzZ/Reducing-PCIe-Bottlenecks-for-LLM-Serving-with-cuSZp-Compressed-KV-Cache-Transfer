"""Select a reproducible mixed-bound candidate from a calibrated profile."""

import argparse
import copy
import json
from pathlib import Path


def select_mixed_profile(profile, *, tight_bound, loose_bound, loose_layers):
    if tight_bound <= 0 or loose_bound <= tight_bound:
        raise ValueError("bounds must satisfy 0 < tight_bound < loose_bound")
    layers = profile.get("layers", {})
    if loose_layers < 0 or loose_layers > len(layers):
        raise ValueError("loose_layers is outside the layer count")
    ranked = []
    for layer_id, entry in layers.items():
        details = {float(item["eps"]): item for item in entry.get("details", [])}
        if tight_bound not in details or loose_bound not in details:
            raise ValueError(f"layer {layer_id} is missing a requested bound")
        tight = details[tight_bound]
        loose = details[loose_bound]
        if not tight.get("safe", True):
            raise ValueError(f"layer {layer_id} is unsafe at the tight bound")
        ranked.append((
            -float(loose["top1_match"]),
            float(loose["kl"]),
            int(layer_id),
        ))
    ranked.sort()
    selected = {layer_id for _, _, layer_id in ranked[:loose_layers]}
    result = copy.deepcopy(profile)
    for layer_id, entry in result["layers"].items():
        entry["max_safe_eps"] = (
            float(loose_bound) if int(layer_id) in selected else float(tight_bound)
        )
    metadata = result.setdefault("_metadata", {})
    metadata.update({
        "method": "ranked_mixed_bound_candidate_for_full_workload_validation",
        "source_method": profile.get("_metadata", {}).get("method"),
        "tight_bound": float(tight_bound),
        "loose_bound": float(loose_bound),
        "loose_layer_count": loose_layers,
        "loose_layers": sorted(selected),
        "ranking": "higher_min_top1_then_lower_worst_kl_then_layer_id",
        "full_workload_validation_required": True,
    })
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tight-bound", type=float, default=1e-5)
    parser.add_argument("--loose-bound", type=float, default=1e-4)
    parser.add_argument("--loose-layers", type=int, required=True)
    args = parser.parse_args()

    profile = json.loads(Path(args.profile).read_text(encoding="utf-8"))
    result = select_mixed_profile(
        profile,
        tight_bound=args.tight_bound,
        loose_bound=args.loose_bound,
        loose_layers=args.loose_layers,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["_metadata"], indent=2))


if __name__ == "__main__":
    main()
