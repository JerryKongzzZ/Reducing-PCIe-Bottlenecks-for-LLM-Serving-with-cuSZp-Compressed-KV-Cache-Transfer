"""Static consistency checks for the reproducibility contract and paper source."""

from __future__ import annotations

import json
from pathlib import Path
import re


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_paper_uses_complete_inline_ieee_bibliography():
    source = (REPO_ROOT / "paper/conference_101719.tex").read_text(
        encoding="utf-8"
    )
    cited = {
        key.strip()
        for group in re.findall(r"\\cite\{([^}]+)\}", source)
        for key in group.split(",")
    }
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", source))

    assert "\\begin{thebibliography}{00}" in source
    assert "\\bibliography{" not in source
    assert "\\bibliographystyle{" not in source
    assert cited == bibliography
    assert len(bibliography) == 14


def test_formal_artifact_roots_are_explicit_and_real():
    config = json.loads(
        (REPO_ROOT / "reproducibility/formal_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    assert config["roots"]
    for relative in config["roots"]:
        assert (REPO_ROOT / relative).is_dir()
        assert "synthetic" not in relative.lower()
        assert "simulation" not in relative.lower()


def test_formal_hash_manifest_is_complete_and_valid():
    config = json.loads(
        (REPO_ROOT / "reproducibility/formal_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    expected = {
        path.relative_to(REPO_ROOT).as_posix()
        for root in config["roots"]
        for path in (REPO_ROOT / root).rglob("*")
        if path.is_file() and path.suffix in config["included_suffixes"]
    }
    expected.update(config["files"])

    manifest = REPO_ROOT / "reproducibility/formal_data.sha256"
    recorded = {
        line.split(maxsplit=1)[1].lstrip("*")
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    assert recorded == expected


def test_run_sh_never_builds_the_pdf():
    source = (REPO_ROOT / "run.sh").read_text(encoding="utf-8")
    assert "pdflatex" not in source
    assert "bibtex" not in source
    for target in (
        "validate", "dry-run", "calibrate", "calibrate-fresh",
        "experiments", "rerun", "assets", "verify-paper",
    ):
        assert target in source


def test_paper_inputs_and_figures_exist_without_archive_dependencies():
    paper = REPO_ROOT / "paper"
    source = (paper / "conference_101719.tex").read_text(encoding="utf-8")
    assert "archive/" not in source
    assert r"\input{" not in source
    assert r"\graphicspath{{./}}" in source

    for relative in re.findall(r"\\input\{([^}]+)\}", source):
        target = paper / relative
        if target.suffix == "":
            target = target.with_suffix(".tex")
        assert target.is_file(), target

    filenames = re.findall(
        r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}",
        source,
    )
    assert set(filenames) == {
        "gate-d-cross-model-handler.png",
        "open-loop-arrival-qwen1p5b.png",
        "open-loop-paired-effects.png",
    }
    for filename in filenames:
        candidates = (
            paper / filename,
            REPO_ROOT / "figures/infocom" / filename,
        )
        assert any(path.is_file() for path in candidates), filename

def test_adaptive_calibration_inputs_are_admitted_and_portable():
    config = json.loads(
        (REPO_ROOT / "reproducibility/formal_artifacts.json").read_text(
            encoding="utf-8"
        )
    )
    expected_root = (
        "data/qwen2.5-1.5b_multi_prompt_sensitivity_1e-5_1e-4_per_prompt"
    )
    assert expected_root in config["roots"]
    assert "data/infocom_calibration_prompts.json" in config["files"]
    assert (
        "data/qwen2.5-1.5b_multi_prompt_sensitivity_top1-0.875.json"
        in config["files"]
    )

    profile = json.loads(
        (
            REPO_ROOT
            / "data/qwen2.5-1.5b_mixed_21x1e-4_7x1e-5.json"
        ).read_text(encoding="utf-8")
    )
    metadata = profile["_metadata"]
    paths = [metadata["prompt_suite"], *metadata["source_profiles"]]
    assert all(not Path(value).is_absolute() for value in paths)
    assert all((REPO_ROOT / value).is_file() for value in paths)


def test_paper_describes_the_implemented_connector_and_evidence_boundary():
    source = (REPO_ROOT / "paper/conference_101719.tex").read_text(
        encoding="utf-8"
    )
    assert "A Python monkey patch" not in source
    assert r"spec\_module\_path" in source
    assert "synchronization-free path" in source
    assert "continuous synthetic" not in source
    assert "cost-aware option disabled" in source
