"""Collect stable source, environment, model, and hardware provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


PACKAGE_NAMES = (
    "torch",
    "numpy",
    "pybind11",
    "vllm",
    "transformers",
    "matplotlib",
    "tqdm",
)
SOURCE_ROOTS = ("benchmarks", "integration", "tests", "reproducibility")
SOURCE_SUFFIXES = {".py", ".cpp", ".h", ".cu", ".txt", ".json", ".sh"}
ROOT_SOURCES = ("run.sh", "test.sh", "requirements.txt", ".gitignore")


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout.strip()


def source_tree_sha256(repo_root: Path) -> str:
    digest = hashlib.sha256()
    paths: list[Path] = []
    for root_name in SOURCE_ROOTS:
        root = repo_root / root_name
        if not root.is_dir():
            continue
        paths.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in SOURCE_SUFFIXES
            and "build_local" not in path.parts
            and "__pycache__" not in path.parts
        )
    paths.extend(
        repo_root / name for name in ROOT_SOURCES if (repo_root / name).is_file()
    )
    for path in sorted(paths, key=lambda item: str(item.relative_to(repo_root))):
        relative = str(path.relative_to(repo_root)).encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


def model_revision(model_id: str) -> str | None:
    cache_root = Path(
        os.environ.get(
            "HF_HUB_CACHE",
            Path.home() / ".cache" / "huggingface" / "hub",
        )
    ).expanduser()
    ref = cache_root / ("models--" + model_id.replace("/", "--")) / "refs" / "main"
    return ref.read_text(encoding="utf-8").strip() if ref.is_file() else None


def collect_runtime_provenance(repo_root: Path, model_id: str) -> dict:
    packages = {}
    for name in PACKAGE_NAMES:
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    git_status = command_output(["git", "status", "--porcelain"], cwd=repo_root)
    gpu = command_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ]
    )
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "command": list(sys.argv),
        "python": platform.python_version(),
        "packages": packages,
        "git_commit": command_output(["git", "rev-parse", "HEAD"], cwd=repo_root),
        "git_dirty": bool(git_status),
        "source_tree_sha256": source_tree_sha256(repo_root),
        "model_id": model_id,
        "model_revision": model_revision(model_id),
        "gpu": gpu,
        "platform": platform.platform(),
    }


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
