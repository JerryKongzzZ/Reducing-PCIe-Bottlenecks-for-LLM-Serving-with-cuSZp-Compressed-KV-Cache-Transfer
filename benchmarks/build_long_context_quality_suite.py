"""Build deterministic long-context needle-retrieval prompts for Gate D."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
RECORDS = (
    ("ZETA-7314", "network telemetry archive"),
    ("NOVA-4826", "compiler validation ledger"),
    ("ORBIT-9053", "marine sensor registry"),
    ("EMBER-2671", "medical imaging catalog"),
    ("LYRIC-6482", "urban rail planning log"),
    ("QUARTZ-1937", "astronomy calibration notebook"),
    ("RAVEN-5148", "datacenter scheduling journal"),
    ("SOLAR-8265", "distributed storage audit"),
)


def build_quality_item(index: int, repeat: int, *, count: int = 8) -> dict:
    """Place one unique answer at a different relative position per request."""
    code, collection = RECORDS[index % len(RECORDS)]
    distractor = (
        f"Request {index} background entry: routine measurements in the "
        f"{collection} were checked and no verification identifier appeared. "
    )
    chunks = [distractor] * int(repeat)
    insertion = max(0, min(len(chunks), ((index + 1) * len(chunks)) // (count + 1)))
    needle = (
        f"IMPORTANT RECORD: the verification code for request {index} is "
        f"{code}. Retain this exact code. "
    )
    chunks.insert(insertion, needle)
    prompt = (
        "Read the archival text and retrieve the requested verification code. "
        "Background entries do not contain the answer.\n\n"
        + "".join(chunks)
        + f"\n\nQuestion: What is the verification code for request {index}? "
        "Answer with the code only."
    )
    return {
        "id": f"needle_{index}",
        "prompt": prompt,
        "expected_answer": code,
        "needle_fraction": insertion / max(int(repeat), 1),
    }


def build_quality_items(repeat: int, *, count: int = 8) -> list[dict]:
    if repeat <= 0:
        raise ValueError("repeat must be positive")
    if count <= 0 or count > len(RECORDS):
        raise ValueError(f"count must be in [1, {len(RECORDS)}]")
    return [
        build_quality_item(index, repeat, count=count)
        for index in range(count)
    ]


def calibrated_quality_repeat(
    tokenizer,
    *,
    max_model_len: int,
    max_tokens: int,
    upper_bound: int,
    count: int = 8,
) -> int:
    """Find the largest common repeat that leaves decode-token headroom."""
    from benchmarks.prompt_workload import build_warmup_prompts

    token_limit = int(max_model_len) - int(max_tokens)
    if token_limit <= 0 or upper_bound <= 0:
        raise ValueError("quality calibration requires a positive token budget")

    def fits(repeat: int) -> bool:
        prompts = [item["prompt"] for item in build_quality_items(repeat, count=count)]
        prompts.extend(build_warmup_prompts(repeat))
        return all(
            len(tokenizer.encode(prompt, add_special_tokens=True)) <= token_limit
            for prompt in prompts
        )

    low, high = 1, int(upper_bound)
    accepted = 0
    while low <= high:
        middle = (low + high) // 2
        if fits(middle):
            accepted = middle
            low = middle + 1
        else:
            high = middle - 1
    if accepted == 0:
        raise ValueError("one quality prompt unit already exceeds the context")
    return accepted


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--repeat-upper-bound", type=int, default=400)
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument(
        "--output",
        default="data/infocom_long_context_quality_v1/prompts.json",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        local_files_only=True,
    )
    repeat = calibrated_quality_repeat(
        tokenizer,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        upper_bound=args.repeat_upper_bound,
        count=args.count,
    )
    items = build_quality_items(repeat, count=args.count)
    output = (REPO_ROOT / args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(items, indent=2), encoding="utf-8")
    token_counts = [
        len(tokenizer.encode(item["prompt"], add_special_tokens=True))
        for item in items
    ]
    metadata = {
        "schema_version": 1,
        "task": "needle_retrieval",
        "model_tokenizer": args.model,
        "count": args.count,
        "repeat": repeat,
        "max_model_len": args.max_model_len,
        "max_tokens": args.max_tokens,
        "prompt_token_counts": token_counts,
        "expected_answers": [item["expected_answer"] for item in items],
        "prompt_file": str(output),
    }
    output.with_suffix(".meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
