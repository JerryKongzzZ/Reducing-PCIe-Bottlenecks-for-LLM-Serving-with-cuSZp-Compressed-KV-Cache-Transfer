"""Generate paper-ready INFOCOM figures from canonical benchmark JSON."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.run_vllm_repeated_smoke import describe


RAW = "async_raw"
CUSZP = "async_cuszp_1e-5"
COLORS = {RAW: "#4C78A8", CUSZP: "#F58518"}
LABELS = {RAW: "Raw", CUSZP: "cuSZp (1e-5)"}


def load_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def arrival_rows(document: dict) -> list[dict]:
    rows = []
    for level in document["levels"]:
        if level.get("status") != "complete":
            continue
        result = level["result"]
        rate = float(result["offered_requests_per_second"])
        for method in (RAW, CUSZP):
            metrics = result["methods"][method]
            latency = metrics["latency"]
            rows.append(
                {
                    "model": document["model"],
                    "offered_rps": rate,
                    "method": method,
                    "achieved_rps": float(
                        metrics["achieved_requests_per_second"]
                    ),
                    "ttft_p95_ms": float(latency["ttft_ms"]["p95"]),
                    "e2e_p95_ms": float(latency["e2e_ms"]["p95"]),
                    "quality_passed": metrics.get("quality_passed"),
                }
            )
    return rows


def open_loop_effect_rows(document: dict) -> list[dict]:
    rows = []
    for level in document["levels"]:
        if level.get("status") != "complete":
            continue
        result = level["result"]
        metrics = result["methods"][CUSZP]
        paired = metrics["paired_vs_raw"]
        handler = metrics["total_handler_vs_raw_ms"]
        e2e = paired["initial_e2e_ms"]
        rows.append(
            {
                "model": document["model"],
                "offered_rps": float(result["offered_requests_per_second"]),
                "handler_diff_ms": float(handler["mean"]),
                "handler_ci95_ms": float(handler["ci95_half_width"]),
                "e2e_diff_ms": float(e2e["mean"]),
                "e2e_ci95_ms": float(e2e["ci95_half_width"]),
                "quality_passed": metrics.get("quality_passed"),
                "trials": int(result["num_trials"]),
            }
        )
    return rows


def gate_d_rows(summary: dict, repo_root: Path = REPO_ROOT) -> list[dict]:
    rows = []
    for model in summary["models"]:
        if model.get("status") != "complete":
            continue
        aggregate = load_json(repo_root / model["aggregate"])
        raw_trials = aggregate["methods"][RAW]["trial_details"]
        cuszp_trials = aggregate["methods"][CUSZP]["trial_details"]
        differences = [
            (
                float(candidate["gpu_to_cpu_ms"])
                + float(candidate["cpu_to_gpu_ms"])
                - float(baseline["gpu_to_cpu_ms"])
                - float(baseline["cpu_to_gpu_ms"])
            )
            for baseline, candidate in zip(raw_trials, cuszp_trials)
        ]
        interval = describe(differences)
        result = model["result"]["methods"][CUSZP]
        rows.append(
            {
                "model": model["model"],
                "handler_diff_ms": float(interval["mean"]),
                "handler_ci95_ms": float(interval["ci95_half_width"]),
                "compression_ratio": float(result["compression_ratio"]),
                "quality_passed": bool(result["quality_passed"]),
                "trials": len(differences),
            }
        )
    return rows


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )


def save_figure(figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        out_dir / f"{stem}.png",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(figure)


def plot_arrival_curve(rows: list[dict], out_dir: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(7.1, 2.35))
    fields = (
        ("achieved_rps", "Achieved throughput (req/s)", "(a) Throughput"),
        ("ttft_p95_ms", "P95 TTFT (ms)", "(b) Tail TTFT"),
        ("e2e_p95_ms", "P95 E2E latency (ms)", "(c) Tail E2E"),
    )
    rates = sorted({row["offered_rps"] for row in rows})
    for axis, (field, ylabel, title) in zip(axes, fields):
        for method, marker in ((RAW, "o"), (CUSZP, "s")):
            points = sorted(
                (row for row in rows if row["method"] == method),
                key=lambda row: row["offered_rps"],
            )
            axis.plot(
                [row["offered_rps"] for row in points],
                [row[field] for row in points],
                marker=marker,
                linewidth=1.4,
                markersize=4,
                color=COLORS[method],
                label=LABELS[method],
            )
        if field == "achieved_rps":
            axis.plot(
                rates,
                rates,
                color="0.45",
                linestyle="--",
                linewidth=1,
                label="Offered = achieved",
            )
        axis.set_xlabel("Offered load (req/s)")
        axis.set_ylabel(ylabel)
        axis.set_title(title, loc="left")
        axis.grid(axis="y", color="0.88", linewidth=0.6)
        axis.set_xticks(rates)
    axes[0].legend(frameon=False, loc="upper left")
    figure.tight_layout(w_pad=1.2)
    save_figure(figure, out_dir, "open-loop-arrival-qwen1p5b")


def short_model_name(model: str) -> str:
    aliases = {
        "Qwen/Qwen2.5-0.5B": "Qwen2.5-0.5B",
        "Qwen/Qwen2.5-1.5B": "Qwen2.5-1.5B",
        "facebook/opt-125m": "OPT-125M",
        "facebook/opt-350m": "OPT-350M",
        "EleutherAI/pythia-160m": "Pythia-160M",
        "EleutherAI/pythia-410m": "Pythia-410M",
        "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T": "TinyLlama-1.1B",
    }
    return aliases.get(model, model)


def plot_open_loop_effects(rows: list[dict], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: (row["model"], row["offered_rps"]))
    labels = [
        f"{short_model_name(row['model'])} @ {row['offered_rps']:g} req/s"
        for row in rows
    ]
    y = list(range(len(rows)))
    figure, axes = plt.subplots(1, 2, figsize=(7.1, 2.65), sharey=True)
    fields = (
        (
            "handler_diff_ms",
            "handler_ci95_ms",
            "Handler difference (ms)",
            "(a) Transfer-path effect",
        ),
        (
            "e2e_diff_ms",
            "e2e_ci95_ms",
            "Initial E2E difference (ms)",
            "(b) Request-level effect",
        ),
    )
    for axis, (field, ci_field, xlabel, title) in zip(axes, fields):
        for index, row in enumerate(rows):
            color = COLORS[CUSZP] if row["quality_passed"] else "0.55"
            axis.errorbar(
                row[field],
                index,
                xerr=row[ci_field],
                fmt="s",
                markersize=4,
                capsize=2.5,
                linewidth=1.1,
                color=color,
            )
        axis.axvline(0, color="0.35", linewidth=1)
        axis.grid(axis="x", color="0.88", linewidth=0.6)
        axis.set_xlabel(xlabel)
        axis.set_title(title, loc="left")
        axis.set_yticks(y, labels)
        axis.invert_yaxis()
    figure.text(
        0.5,
        0.01,
        "cuSZp - raw; negative is better. Error bars are paired 95% CIs.",
        ha="center",
        fontsize=7,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 1), w_pad=1.4)
    save_figure(figure, out_dir, "open-loop-paired-effects")


def plot_gate_d(rows: list[dict], out_dir: Path) -> None:
    rows = sorted(rows, key=lambda row: row["handler_diff_ms"])
    y = list(range(len(rows)))
    figure, axis = plt.subplots(figsize=(5.0, 3.2))
    for index, row in enumerate(rows):
        color = COLORS[CUSZP] if row["quality_passed"] else "0.55"
        marker = "s" if row["quality_passed"] else "x"
        axis.errorbar(
            row["handler_diff_ms"],
            index,
            xerr=row["handler_ci95_ms"],
            fmt=marker,
            markersize=5,
            capsize=2.5,
            linewidth=1.1,
            color=color,
        )
    axis.axvline(0, color="0.35", linewidth=1)
    axis.grid(axis="x", color="0.88", linewidth=0.6)
    axis.set_yticks(y, [short_model_name(row["model"]) for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel("Handler difference, cuSZp - raw (ms)")
    axis.set_title("Gate D cross-model fixed-bound effect", loc="left")
    axis.text(
        0.99,
        0.01,
        "square: quality pass; x: quality reject",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    figure.tight_layout()
    save_figure(figure, out_dir, "gate-d-cross-model-handler")


def latex_escape(value: str) -> str:
    return (
        value.replace("\\", r"\textbackslash{}")
        .replace("&", r"\&")
        .replace("%", r"\%")
        .replace("_", r"\_")
    )


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_table_exports(
    out_dir: Path,
    open_loop_rows: list[dict],
    gate_rows: list[dict],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    effect_fields = [
        "model",
        "offered_rps",
        "handler_diff_ms",
        "handler_ci95_ms",
        "e2e_diff_ms",
        "e2e_ci95_ms",
        "quality_passed",
        "trials",
    ]
    gate_fields = [
        "model",
        "handler_diff_ms",
        "handler_ci95_ms",
        "compression_ratio",
        "quality_passed",
        "trials",
    ]
    write_csv(out_dir / "open-loop-paired-effects.csv", open_loop_rows, effect_fields)
    write_csv(out_dir / "gate-d-cross-model-handler.csv", gate_rows, gate_fields)

    effect_lines = [
        r"\begin{tabular}{lrrrrc}",
        r"\toprule",
        r"Model & Load & Handler $\Delta$ & 95\% CI & E2E $\Delta$ & Quality \\",
        r"\midrule",
    ]
    for row in open_loop_rows:
        effect_lines.append(
            "{} & {:.0f} & {:.3f} & +/- {:.3f} & {:.3f} & {} \\\\".format(
                latex_escape(short_model_name(row["model"])),
                row["offered_rps"],
                row["handler_diff_ms"],
                row["handler_ci95_ms"],
                row["e2e_diff_ms"],
                "pass" if row["quality_passed"] else "reject",
            )
        )
    effect_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out_dir / "open-loop-paired-effects.tex").write_text(
        "\n".join(effect_lines), encoding="utf-8"
    )

    gate_lines = [
        r"\begin{tabular}{lrrrc}",
        r"\toprule",
        r"Model & Handler $\Delta$ & 95\% CI & Ratio & Quality \\",
        r"\midrule",
    ]
    for row in gate_rows:
        gate_lines.append(
            "{} & {:.3f} & +/- {:.3f} & {:.3f}x & {} \\\\".format(
                latex_escape(short_model_name(row["model"])),
                row["handler_diff_ms"],
                row["handler_ci95_ms"],
                row["compression_ratio"],
                "pass" if row["quality_passed"] else "reject",
            )
        )
    gate_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    (out_dir / "gate-d-cross-model-handler.tex").write_text(
        "\n".join(gate_lines), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--qwen1p5-arrival",
        type=Path,
        default=REPO_ROOT
        / "data/qwen1.5b_open_loop_arrival_v1/arrival_rate_summary.json",
    )
    parser.add_argument(
        "--qwen0p5-arrival",
        type=Path,
        default=REPO_ROOT
        / "data/qwen0.5b_open_loop_arrival_v1/arrival_rate_summary.json",
    )
    parser.add_argument(
        "--gate-d",
        type=Path,
        default=REPO_ROOT / "data/gate_d_fair_async_probe/gate_d_summary.json",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=REPO_ROOT / "figures/infocom"
    )
    args = parser.parse_args()

    qwen1p5 = load_json(args.qwen1p5_arrival)
    qwen0p5 = load_json(args.qwen0p5_arrival)
    gate_d = load_json(args.gate_d)
    configure_style()
    arrival = arrival_rows(qwen1p5)
    effects = open_loop_effect_rows(qwen0p5) + open_loop_effect_rows(qwen1p5)
    cross_model = gate_d_rows(gate_d)
    plot_arrival_curve(arrival, args.out_dir)
    plot_open_loop_effects(effects, args.out_dir)
    plot_gate_d(cross_model, args.out_dir)
    write_table_exports(args.out_dir, effects, cross_model)
    print(args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
