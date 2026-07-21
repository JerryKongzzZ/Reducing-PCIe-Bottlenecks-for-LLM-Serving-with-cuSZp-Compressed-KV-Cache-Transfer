from pathlib import Path
import sys

from benchmarks import generate_paper_assets as paper_assets


ROOT = Path(__file__).resolve().parents[1]


def test_current_canonical_metrics_are_rendered() -> None:
    concurrency = paper_assets.load(
        ROOT / "data/qwen1.5b_real_concurrency_v3/concurrency_summary.json"
    )
    qwen1p5 = paper_assets.load(
        ROOT / "data/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json"
    )
    qwen0p5 = paper_assets.load(
        ROOT / "data/qwen0.5b_open_loop_arrival_v1/arrival_rate_summary.json"
    )

    macros = paper_assets.result_macros(concurrency, qwen1p5, qwen0p5)

    assert r"\ConcurrencyHandlerReductionPct}{25.5\%}" in macros
    assert r"\ConcurrencyThroughputGainPct}{20.6\%}" in macros
    assert r"\ArrivalAchievedGainPct}{17.7\%}" in macros
    assert r"\ArrivalTTFTPReductionPct}{38.4\%}" in macros
    assert "\\C8" not in macros
    assert "\\R6" not in macros


def test_generator_writes_only_current_project_tables(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys, "argv", ["generate_paper_assets.py", "--out-dir", str(tmp_path)]
    )

    assert paper_assets.main() == 0

    names = {path.name for path in tmp_path.iterdir()}
    assert names == {
        "results_macros.tex",
        "table_adaptive.tex",
        "table_concurrency.tex",
        "table_gate_d.tex",
        "table_open_loop.tex",
        "table_quality.tex",
    }
    open_loop = (tmp_path / "table_open_loop.tex").read_text(encoding="utf-8")
    assert "7.920 $\\rightarrow$ 7.981" in open_loop
    assert "3.751 $\\rightarrow$ 4.415" in open_loop
    assert "five trials" in open_loop

def test_standalone_paper_blocks_match_canonical_json() -> None:
    concurrency = paper_assets.load(
        ROOT / "data/qwen1.5b_real_concurrency_v3/concurrency_summary.json"
    )
    qwen1p5 = paper_assets.load(
        ROOT / "data/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json"
    )
    qwen0p5 = paper_assets.load(
        ROOT / "data/qwen0.5b_open_loop_arrival_v1/arrival_rate_summary.json"
    )
    assets = {
        "results_macros": paper_assets.result_macros(
            concurrency, qwen1p5, qwen0p5
        ),
        "table_concurrency": paper_assets.concurrency_table(concurrency),
        "table_open_loop": paper_assets.open_loop_table([qwen0p5, qwen1p5]),
        "table_gate_d": paper_assets.gate_d_table(
            paper_assets.load(
                ROOT / "data/gate_d_fair_async_probe/gate_d_summary.json"
            )
        ),
        "table_quality": paper_assets.quality_table(
            paper_assets.load(
                ROOT
                / "data/infocom_long_context_quality_v1/five_trial/aggregate.json"
            )
        ),
        "table_adaptive": paper_assets.adaptive_table(
            paper_assets.load(
                ROOT
                / "data/vllm_qwen1.5b_4k_gate_c_packed_mixed21_complete_2trial/aggregate.json"
            )
        ),
    }
    changed = paper_assets.sync_paper_blocks(
        ROOT / "paper/conference_101719.tex", assets, check_only=True
    )
    assert changed is False
