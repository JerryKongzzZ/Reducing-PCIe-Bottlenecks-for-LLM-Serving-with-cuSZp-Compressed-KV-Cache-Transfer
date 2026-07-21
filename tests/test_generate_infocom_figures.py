import json


def test_arrival_rows_and_effects_preserve_canonical_metrics():
    from benchmarks.generate_infocom_figures import (
        arrival_rows,
        open_loop_effect_rows,
    )

    method = {
        "achieved_requests_per_second": 3.0,
        "latency": {
            "ttft_ms": {"p95": 100.0},
            "e2e_ms": {"p95": 200.0},
        },
        "quality_passed": True,
    }
    cuszp = {
        **method,
        "total_handler_vs_raw_ms": {
            "mean": -2.0,
            "ci95_half_width": 0.5,
        },
        "paired_vs_raw": {
            "initial_e2e_ms": {
                "mean": -10.0,
                "ci95_half_width": 3.0,
            }
        },
    }
    document = {
        "model": "example/model",
        "levels": [
            {
                "status": "complete",
                "result": {
                    "num_trials": 5,
                    "offered_requests_per_second": 4.0,
                    "methods": {
                        "async_raw": method,
                        "async_cuszp_1e-5": cuszp,
                    },
                },
            },
            {"status": "failed"},
        ],
    }

    rows = arrival_rows(document)
    effects = open_loop_effect_rows(document)
    assert len(rows) == 2
    assert rows[1]["ttft_p95_ms"] == 100.0
    assert effects == [
        {
            "model": "example/model",
            "offered_rps": 4.0,
            "handler_diff_ms": -2.0,
            "handler_ci95_ms": 0.5,
            "e2e_diff_ms": -10.0,
            "e2e_ci95_ms": 3.0,
            "quality_passed": True,
            "trials": 5,
        }
    ]


def test_gate_d_rows_computes_paired_handler_interval(tmp_path):
    from benchmarks.generate_infocom_figures import gate_d_rows

    aggregate = {
        "methods": {
            "async_raw": {
                "trial_details": [
                    {"gpu_to_cpu_ms": 10.0, "cpu_to_gpu_ms": 2.0},
                    {"gpu_to_cpu_ms": 12.0, "cpu_to_gpu_ms": 2.0},
                ]
            },
            "async_cuszp_1e-5": {
                "trial_details": [
                    {"gpu_to_cpu_ms": 8.0, "cpu_to_gpu_ms": 1.0},
                    {"gpu_to_cpu_ms": 9.0, "cpu_to_gpu_ms": 2.0},
                ]
            },
        }
    }
    path = tmp_path / "aggregate.json"
    path.write_text(json.dumps(aggregate), encoding="utf-8")
    summary = {
        "models": [
            {
                "model": "example/model",
                "status": "complete",
                "aggregate": "aggregate.json",
                "result": {
                    "methods": {
                        "async_cuszp_1e-5": {
                            "compression_ratio": 1.5,
                            "quality_passed": True,
                        }
                    }
                },
            }
        ]
    }

    row = gate_d_rows(summary, tmp_path)[0]
    assert row["handler_diff_ms"] == -3.0
    assert row["handler_ci95_ms"] == 0.0
    assert row["quality_passed"] is True
    assert row["trials"] == 2

def test_table_exports_are_machine_and_latex_readable(tmp_path):
    from benchmarks.generate_infocom_figures import write_table_exports

    effects = [
        {
            "model": "Qwen/Qwen2.5-0.5B",
            "offered_rps": 8.0,
            "handler_diff_ms": -10.0,
            "handler_ci95_ms": 1.0,
            "e2e_diff_ms": -30.0,
            "e2e_ci95_ms": 5.0,
            "quality_passed": True,
            "trials": 5,
        }
    ]
    gate = [
        {
            "model": "facebook/opt-350m",
            "handler_diff_ms": -3.0,
            "handler_ci95_ms": 1.0,
            "compression_ratio": 1.2,
            "quality_passed": False,
            "trials": 5,
        }
    ]
    write_table_exports(tmp_path, effects, gate)

    csv_text = (tmp_path / "open-loop-paired-effects.csv").read_text()
    tex_text = (tmp_path / "gate-d-cross-model-handler.tex").read_text()
    assert "handler_diff_ms" in csv_text
    assert "Qwen/Qwen2.5-0.5B" in csv_text
    assert r"\begin{tabular}" in tex_text
    assert "OPT-350M" in tex_text

def test_save_figure_emits_png_only(tmp_path):
    import matplotlib.pyplot as plt

    from benchmarks.generate_infocom_figures import save_figure

    figure, axis = plt.subplots()
    axis.plot([0, 1], [0, 1])
    save_figure(figure, tmp_path, "example")

    assert (tmp_path / "example.png").is_file()
    assert not (tmp_path / "example.pdf").exists()

