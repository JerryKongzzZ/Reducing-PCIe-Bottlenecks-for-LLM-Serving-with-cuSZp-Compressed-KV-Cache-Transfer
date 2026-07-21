"""Versioned prompt construction and tokenizer-aware workload sizing.

``legacy_disjoint_v1`` reproduces the original Gate D/adaptive workload in
which streams 6 and 7 reuse the seed text of streams 0 and 1. ``disjoint`` is
the corrected v2 workload: every request carries an explicit stream id and is
therefore unique even after the six seed families wrap. Keep both names stable
because archived paper results depend on v1 while new experiments use v2.
"""


BASE_PROMPT = "KV cache offloading over PCIe is useful because "
DISJOINT_SEEDS = (
    "Network packet scheduling reduces congestion in datacenter switches. ",
    "Astronomers measure distant galaxies using calibrated spectral sensors. ",
    "Marine ecosystems depend on stable temperature and nutrient cycles. ",
    "Compiler optimization transforms programs while preserving exact semantics. ",
    "Urban rail planning balances passenger demand and station capacity. ",
    "Medical imaging reconstructs internal structures from noisy measurements. ",
)


def build_pressure_prompts(repeats, *, style="disjoint"):
    prompts = []
    for index, repeat in enumerate(repeats):
        if style == "shared":
            unit = BASE_PROMPT + f"experiment stream {index}. "
        elif style == "legacy_disjoint_v1":
            unit = DISJOINT_SEEDS[index % len(DISJOINT_SEEDS)]
        elif style == "disjoint":
            unit = (
                f"Independent request stream {index}. "
                + DISJOINT_SEEDS[index % len(DISJOINT_SEEDS)]
            )
        else:
            raise ValueError(f"unknown prompt workload style: {style}")
        prompts.append(unit * int(repeat))
    return prompts


def build_warmup_prompts(repeat, count=6):
    return [
        (BASE_PROMPT + f"warmup stream {index}. ") * int(repeat)
        for index in range(count)
    ]


def calibrated_repeat(
    tokenizer,
    *,
    max_model_len,
    max_tokens,
    upper_bound,
    prompt_count=8,
    style="disjoint",
):
    """Largest repeat up to ``upper_bound`` that leaves decode headroom."""
    token_limit = int(max_model_len) - int(max_tokens)
    if token_limit <= 0 or upper_bound <= 0:
        raise ValueError("prompt calibration requires a positive token budget")

    def fits(repeat):
        prompts = build_pressure_prompts(
            [repeat] * prompt_count, style=style
        )
        prompts.extend(build_warmup_prompts(repeat))
        return all(
            len(tokenizer.encode(prompt, add_special_tokens=True))
            <= token_limit
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
        raise ValueError("one prompt unit already exceeds the model context")
    return accepted
