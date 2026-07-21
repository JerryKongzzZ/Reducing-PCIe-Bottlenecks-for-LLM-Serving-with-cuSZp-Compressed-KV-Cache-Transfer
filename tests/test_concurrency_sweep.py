def metric(mean):
    return {"mean": mean}


def method_metrics(g2c, c2g, *, ratio=1.0, quality=1.0):
    return {
        "compression_ratio": metric(ratio),
        "gpu_to_cpu_ms": metric(g2c),
        "cpu_to_gpu_ms": metric(c2g),
        "initial_e2e_ms": metric(100.0),
        "replay_e2e_ms": metric(50.0),
        "replay_ttft_ms": metric(40.0),
        "token_match_rate": metric(quality),
        "exact_match_rate": metric(quality),
        "initial_requests_per_second": metric(8.0),
        "replay_requests_per_second": metric(10.0),
        "request_latency_distribution": {
            "initial": {
                "count": 10,
                "ttft_ms": {"p95": 900.0, "p99": 990.0},
                "e2e_ms": {"p95": 1100.0, "p99": 1190.0},
                "slo": {
                    "ttft_ms": {"attainment": 1.0},
                    "e2e_ms": {"attainment": 0.9},
                },
            }
        },
        "trial_details": [
            {"gpu_to_cpu_ms": g2c, "cpu_to_gpu_ms": c2g},
            {"gpu_to_cpu_ms": g2c + 1.0, "cpu_to_gpu_ms": c2g},
        ],
    }


def test_concurrency_summary_builds_paired_total_handler_difference():
    from benchmarks.run_concurrency_sweep import summarize_aggregate

    document = {
        "model": "example/model",
        "num_trials": 2,
        "prompt_repeats": [32] * 8,
        "burst_slo_thresholds_ms": {"ttft": 2000.0, "e2e": 2500.0},
        "methods": {
            "async_raw": method_metrics(10.0, 2.0),
            "async_cuszp_1e-5": method_metrics(
                8.0, 1.0, ratio=1.5
            ),
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
    assert result["prompt_count"] == 8
    assert cuszp["total_handler_ms"] == 9.0
    assert cuszp["total_handler_vs_raw_ms"]["mean"] == -3.0
    assert cuszp["initial_requests_per_second"] == 8.0
    assert cuszp["request_latency_distribution"]["initial"]["count"] == 10
    assert result["burst_slo_thresholds_ms"]["ttft"] == 2000.0
    assert cuszp["quality_passed"]


def test_concurrency_report_states_no_synthetic_contender():
    from benchmarks.run_concurrency_sweep import render_markdown

    report = render_markdown({"levels": []})

    assert "No synthetic PCIe copy contender" in report
    assert "unique" in report
    assert "Pooled burst tail latency" in report
    assert "open-loop" in report
