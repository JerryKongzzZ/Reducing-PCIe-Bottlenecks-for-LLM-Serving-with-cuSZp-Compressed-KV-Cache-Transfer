import pytest


class WordTokenizer:
    def encode(self, text, add_special_tokens=True):
        tokens = text.split()
        return ([0] if add_special_tokens else []) + tokens


def test_quality_items_are_unique_and_cover_needle_positions():
    from benchmarks.build_long_context_quality_suite import build_quality_items

    items = build_quality_items(90, count=8)

    assert len({item["prompt"] for item in items}) == 8
    assert len({item["expected_answer"] for item in items}) == 8
    assert all(
        item["expected_answer"] in item["prompt"]
        for item in items
    )
    positions = [item["needle_fraction"] for item in items]
    assert positions == sorted(positions)
    assert positions[0] < 0.2
    assert positions[-1] > 0.8


def test_quality_repeat_respects_common_context_limit():
    from benchmarks.build_long_context_quality_suite import (
        build_quality_items,
        calibrated_quality_repeat,
    )
    from benchmarks.prompt_workload import build_warmup_prompts

    tokenizer = WordTokenizer()
    repeat = calibrated_quality_repeat(
        tokenizer,
        max_model_len=180,
        max_tokens=12,
        upper_bound=100,
        count=4,
    )
    prompts = [item["prompt"] for item in build_quality_items(repeat, count=4)]
    prompts.extend(build_warmup_prompts(repeat))
    assert all(len(tokenizer.encode(prompt)) <= 168 for prompt in prompts)


def test_quality_items_validate_requested_count():
    from benchmarks.build_long_context_quality_suite import build_quality_items

    with pytest.raises(ValueError, match="count"):
        build_quality_items(10, count=9)
