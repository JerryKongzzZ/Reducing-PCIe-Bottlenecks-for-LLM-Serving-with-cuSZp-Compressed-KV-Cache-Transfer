"""Merge method results from repeated-smoke aggregates with matching workloads."""

import argparse
import json
from pathlib import Path


CONFIG_KEYS = (
    "schema_version",
    "model",
    "num_trials",
    "warmup_offload",
    "max_model_len",
    "kv_cache_memory_bytes",
    "prompt_repeats",
    "prompt_style",
    "batch_prompts",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--replace-duplicates",
        action="store_true",
        help="Let a method in a later input replace the earlier result.",
    )
    args = parser.parse_args()

    sources = [Path(value) for value in args.inputs]
    documents = [json.loads(path.read_text(encoding="utf-8")) for path in sources]
    reference = documents[0]
    merged_methods = {}
    for path, document in zip(sources, documents):
        for key in CONFIG_KEYS:
            if document.get(key) != reference.get(key):
                raise ValueError(f"workload mismatch for {key!r} in {path}")
        for method, result in document["methods"].items():
            if method in merged_methods:
                if not args.replace_duplicates:
                    raise ValueError(f"duplicate method {method!r} in {path}")
            merged_methods[method] = result

    output = {key: reference.get(key) for key in CONFIG_KEYS}
    output["source_aggregates"] = [str(path) for path in sources]
    output["methods"] = merged_methods
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(destination)


if __name__ == "__main__":
    main()
