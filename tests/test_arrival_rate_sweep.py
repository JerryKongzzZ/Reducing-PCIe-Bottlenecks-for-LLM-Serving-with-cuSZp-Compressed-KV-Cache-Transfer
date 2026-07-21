def metric(mean):
    return {"mean": mean}


def method_metrics(g2c, c2g, *, ratio=1.0):
    latency = {
        "count": 20,
        "ttft_ms": {"p95": 400.0, "p99": 450.0},
        "e2e_ms": {"p95": 600.0, "p99": 700.0},
        "slo": {
            "ttft_ms": {"attainment": 0.95},
            "e2e_ms": {"attainment": 0.90},
        },
    }
    return {
        "compression_ratio": metric(ratio),
        "gpu_to_cpu_ms": metric(g2c),
        "cpu_to_gpu_ms": metric(c2g),
        "initial_requests_per_second": metric(3.5),
        "request_latency_distribution": {"initial": latency},
        "trial_details": [
            {"gpu_to_cpu_ms": g2c, "cpu_to_gpu_ms": c2g},
            {"gpu_to_cpu_ms": g2c + 1.0, "cpu_to_gpu_ms": c2g},
        ],
    }


def test_arrival_summary_preserves_open_loop_latency_and_pairing():
    from benchmarks.run_arrival_rate_sweep import summarize_aggregate

    document = {
        "model": "example/model",
        "num_trials": 2,
        "prompt_repeats": [32] * 10,
        "interarrival_ms": 250.0,
        "burst_slo_thresholds_ms": {"ttft": 500.0, "e2e": 750.0},
        "methods": {
            "async_raw": method_metrics(10.0, 2.0),
            "async_cuszp_1e-5": method_metrics(8.0, 1.0, ratio=1.5),
        },
        "paired_comparisons": {
            "async_cuszp_1e-5": {"metrics": {}},
        },
        "quality_gate": {
            "passed": True,
            "methods": {
                "async_raw": {"passed": True},
                "async_cuszp_1e-5": {"passed": True},
            },
        },
    }

    result = summarize_aggregate(document)
    cuszp = result["methods"]["async_cuszp_1e-5"]
    assert result["offered_requests_per_second"] == 4.0
    assert result["request_count"] == 10
    assert cuszp["latency"]["ttft_ms"]["p95"] == 400.0
    assert cuszp["total_handler_vs_raw_ms"]["mean"] == -3.0
    assert cuszp["quality_passed"]


def test_arrival_report_identifies_open_loop_protocol():
    from benchmarks.run_arrival_rate_sweep import render_markdown

    report = render_markdown({"trials": 2, "levels": []})
    assert "open-loop" in report
    assert "scheduled arrival timestamp" in report
    assert "2-trial paired differences" in report


def test_rate_slug_is_stable_for_fractional_rates():
    from benchmarks.run_arrival_rate_sweep import rate_slug

    assert rate_slug(4.0) == "r4"
    assert rate_slug(2.5) == "r2p5"
