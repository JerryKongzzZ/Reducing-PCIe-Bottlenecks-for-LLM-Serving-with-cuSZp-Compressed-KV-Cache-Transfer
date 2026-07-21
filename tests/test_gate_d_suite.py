def test_gate_d_manifest_contains_eight_unique_cached_model_names():
    from benchmarks.run_gate_d_suite import MODEL_WORKLOADS

    assert len(MODEL_WORKLOADS) == 8
    assert len({item.slug for item in MODEL_WORKLOADS.values()}) == 8
    assert all(item.max_model_len >= 1024 for item in MODEL_WORKLOADS.values())


def test_gate_d_summary_preserves_quality_rejections():
    from benchmarks.run_gate_d_suite import render_markdown, summarize_aggregate

    metric = lambda value: {"mean": value}
    document = {
        "model": "example/model",
        "num_trials": 2,
        "methods": {
            "async_raw": {
                "compression_ratio": metric(1.0),
                "gpu_to_cpu_ms": metric(10.0),
                "cpu_to_gpu_ms": metric(5.0),
                "token_match_rate": metric(1.0),
                "exact_match_rate": metric(1.0),
            },
            "async_int8": {
                "compression_ratio": metric(2.0),
                "gpu_to_cpu_ms": metric(8.0),
                "cpu_to_gpu_ms": metric(4.0),
                "token_match_rate": metric(0.75),
                "exact_match_rate": metric(0.5),
            },
        },
        "quality_gate": {
            "passed": False,
            "methods": {
                "async_raw": {"passed": True},
                "async_int8": {"passed": False},
            },
        },
    }

    result = summarize_aggregate(document)
    assert result["methods"]["async_raw"]["quality_passed"]
    assert not result["methods"]["async_int8"]["quality_passed"]
    markdown = render_markdown({
        "models": [{
            "model": "example/model",
            "status": "complete",
            "result": result,
        }]
    })
    assert "| async_int8 |" in markdown
    assert "reject" in markdown


def test_gate_d_resume_replaces_one_model_without_losing_others():
    from benchmarks.run_gate_d_suite import upsert_model_result

    models = [
        {"model": "first", "status": "complete"},
        {"model": "second", "status": "dry_run"},
    ]
    upsert_model_result(models, {"model": "second", "status": "complete"})

    assert models == [
        {"model": "first", "status": "complete"},
        {"model": "second", "status": "complete"},
    ]


def test_prompt_repeat_calibration_leaves_decode_headroom():
    from benchmarks.prompt_workload import calibrated_repeat

    class CharacterTokenizer:
        def encode(self, text, add_special_tokens=True):
            special = 1 if add_special_tokens else 0
            return list(range(len(text) + special))

    tokenizer = CharacterTokenizer()
    repeat = calibrated_repeat(
        tokenizer,
        max_model_len=256,
        max_tokens=8,
        upper_bound=20,
    )

    assert 0 < repeat < 20

def test_disjoint_pressure_prompts_are_unique_above_seed_count():
    from benchmarks.prompt_workload import build_pressure_prompts

    prompts = build_pressure_prompts(
        [2] * 32,
        style="disjoint",
    )

    assert len(set(prompts)) == len(prompts)


def test_legacy_gate_d_prompts_reproduce_six_seed_eight_slot_protocol():
    from benchmarks.prompt_workload import build_pressure_prompts

    prompts = build_pressure_prompts(
        [2] * 8,
        style="legacy_disjoint_v1",
    )

    assert len(set(prompts)) == 6
    assert prompts[0] == prompts[6]
    assert prompts[1] == prompts[7]


def test_gate_d_runner_out_dir_accepts_external_scratch_path(tmp_path):
    from benchmarks.run_gate_d_suite import runner_out_dir

    assert runner_out_dir(tmp_path) == str(tmp_path)


def test_unversioned_gate_d_summary_is_legacy_and_cannot_be_relabelled():
    from benchmarks.run_gate_d_suite import (
        PAPER_PROTOCOL,
        existing_workload_version,
    )

    assert existing_workload_version({}) == PAPER_PROTOCOL == "legacy_v1"
    assert existing_workload_version({"workload_version": "corrected_v2"}) == (
        "corrected_v2"
    )
