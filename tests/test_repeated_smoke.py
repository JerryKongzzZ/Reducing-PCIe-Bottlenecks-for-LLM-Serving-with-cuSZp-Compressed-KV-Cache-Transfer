import pytest
from types import SimpleNamespace


def test_small_sample_confidence_interval_uses_student_t():
    from benchmarks.run_vllm_repeated_smoke import ci95, describe

    values = [1.0, 2.0, 3.0, 4.0, 5.0]
    expected = 2.776 * 1.5811388300841898 / (5 ** 0.5)
    assert ci95(values) == pytest.approx(expected)
    summary = describe(values)
    assert summary["mean"] == 3.0
    assert summary["trials"] == values


def test_interleaved_trial_schedule_pairs_methods_per_trial():
    from benchmarks.run_vllm_repeated_smoke import interleaved_trial_schedule

    assert interleaved_trial_schedule(("raw", "cuszp"), 3) == [
        ("raw", 1),
        ("cuszp", 1),
        ("raw", 2),
        ("cuszp", 2),
        ("raw", 3),
        ("cuszp", 3),
    ]


def test_method_error_bound_is_only_attached_to_static_cuszp():
    from benchmarks.run_vllm_repeated_smoke import method_error_bound

    assert method_error_bound("async_raw", 1e-4) is None
    assert method_error_bound("async_lz4", 1e-4) is None
    assert method_error_bound("async_adaptive", 1e-4) is None
    assert method_error_bound("async_cuszp", 1e-4) == 1e-4
    assert method_error_bound("async_cuszp_1e-5", 1e-4) == 1e-5


def test_paired_descriptions_reports_candidate_minus_baseline():
    from benchmarks.run_vllm_repeated_smoke import paired_descriptions

    baseline = [{"latency_ms": 5.0}, {"latency_ms": 7.0}]
    candidate = [{"latency_ms": 4.0}, {"latency_ms": 8.0}]
    paired = paired_descriptions(
        baseline,
        candidate,
        ("latency_ms", "missing_metric"),
    )

    assert paired["latency_ms"]["mean"] == 0.0
    assert paired["latency_ms"]["trials"] == [-1.0, 1.0]
    assert "missing_metric" not in paired


def test_quality_gate_rejects_any_trial_below_raw_baseline():
    from benchmarks.run_vllm_repeated_smoke import evaluate_quality_gate

    results = {
        "async_raw": {
            "trial_details": [
                {"token_match_rate": 0.875, "exact_match_rate": 0.875},
                {"token_match_rate": 1.0, "exact_match_rate": 1.0},
            ]
        },
        "async_adaptive": {
            "trial_details": [
                {"token_match_rate": 0.875, "exact_match_rate": 0.875},
                {"token_match_rate": 0.75, "exact_match_rate": 0.75},
            ]
        },
    }

    gate = evaluate_quality_gate(results, baseline_method="async_raw")

    assert not gate["passed"]
    assert gate["methods"]["async_raw"]["passed"]
    assert not gate["methods"]["async_adaptive"]["passed"]
    assert gate["failures"][0]["method"] == "async_adaptive"
    assert gate["failures"][0]["trial"] == 2


def test_request_timing_uses_vllm_timestamp_boundaries():
    from benchmarks.smoke_vllm_compressed_offload import request_timing

    request = SimpleNamespace(
        metrics=SimpleNamespace(
            num_generation_tokens=3,
            arrival_time=10.0,
            scheduled_ts=10.01,
            first_token_ts=10.04,
            last_token_ts=10.10,
            first_token_latency=0.04,
        )
    )
    timing = request_timing(request)
    assert timing["ttft_ms"] == pytest.approx(40.0)
    assert timing["e2e_ms"] == pytest.approx(100.0)
    assert timing["tpot_ms"] == pytest.approx(30.0)


def test_disabled_h2d_contender_is_a_noop():
    from benchmarks.smoke_vllm_compressed_offload import H2DContender

    contender = H2DContender(0, idle_us=125)
    contender.start()
    stats = contender.stop()

    assert stats["size_bytes"] == 0
    assert stats["idle_us"] == 125
    assert stats["bytes_copied"] == 0
    assert stats["throughput_gbps"] == 0.0


def test_trial_metrics_reports_profiled_restore_stages(tmp_path):
    import json

    from benchmarks.run_vllm_repeated_smoke import trial_metrics

    metrics_path = tmp_path / "metrics.jsonl"
    events = [
        {
            "success": True,
            "direction": "gpu_to_cpu",
            "compression_ratio": 2.0,
            "elapsed_seconds": 0.01,
        },
        {
            "success": True,
            "direction": "cpu_to_gpu",
            "original_bytes": 2_000_000,
            "transferred_bytes": 1_000_000,
            "elapsed_seconds": 0.02,
            "effective_h2d_gbps": 8.0,
            "restore_stages": {
                "cpu_decode_seconds": 0.001,
                "h2d_seconds": 0.010,
                "gpu_decode_seconds": 0.003,
                "scatter_seconds": 0.002,
            },
        },
        {
            "success": True,
            "direction": "cpu_to_gpu",
            "original_bytes": 8_000_000,
            "transferred_bytes": 4_000_000,
            "elapsed_seconds": 0.01,
            "effective_h2d_gbps": 999.0,
            "restore_stages": {
                "cpu_decode_seconds": 0.0,
                "h2d_seconds": 0.002,
                "gpu_decode_seconds": 0.001,
                "scatter_seconds": 0.001,
            },
        },
    ]
    metrics_path.write_text(
        "\n".join(json.dumps(event) for event in events), encoding="utf-8"
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        json.dumps({
            "error_bound": 1e-5,
            "mean_token_match_rate": 1.0,
            "exact_match_rate": 1.0,
            "mean_initial_e2e_ms": 10.0,
            "mean_replay_e2e_ms": 8.0,
            "mean_replay_ttft_ms": 5.0,
            "mean_replay_tpot_ms": 1.0,
        }),
        encoding="utf-8",
    )

    result = trial_metrics(metrics_path, summary_path)
    assert result["restore_h2d_ms"] == pytest.approx(6.0)
    assert result["restore_gpu_decode_ms"] == pytest.approx(2.0)
    assert result["restore_profiled_total_ms"] == pytest.approx(10.0)
    assert result["restore_h2d_fraction"] == pytest.approx(0.4)
    assert result["effective_h2d_gbps"] == pytest.approx(10 / 3)
    assert result["mean_restore_original_bytes"] == 5_000_000
    assert result["mean_restore_transferred_bytes"] == 2_500_000
    assert result["restore_decompression_gbps"] == pytest.approx(20.0)


def test_partition_prompt_batches_supports_explicit_pressure_phases():
    from benchmarks.smoke_vllm_compressed_offload import partition_prompt_batches

    prompts = list("abcdefgh")
    assert partition_prompt_batches(prompts, batch_sizes=(1, 1, 6)) == [
        ["a"],
        ["b"],
        list("cdefgh"),
    ]
    with pytest.raises(ValueError, match="sum"):
        partition_prompt_batches(prompts, batch_sizes=(1, 6))
