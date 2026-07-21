#!/usr/bin/env python3
"""Verify the pinned paper environment without running model inference."""

from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)
    print(f"ERROR: {message}", file=sys.stderr)


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def verify_hash_manifest(path: Path, errors: list[str]) -> None:
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split(maxsplit=1)
        except ValueError:
            fail(f"{path}:{line_number}: invalid SHA-256 line", errors)
            continue
        relative = relative.lstrip("*")
        target = REPO_ROOT / relative
        if not target.is_file():
            fail(f"missing hashed artifact: {relative}", errors)
            continue
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            fail(f"hash mismatch: {relative}", errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scope", choices=("software", "full"), default="software"
    )
    parser.add_argument("--check-results", action="store_true")
    parser.add_argument(
        "--allow-different-hardware",
        action="store_true",
        help="Skip exact RTX 5080 and driver matching.",
    )
    args = parser.parse_args()

    environment = json.loads(
        (REPO_ROOT / "reproducibility/environment.json").read_text(
            encoding="utf-8"
        )
    )
    models = json.loads(
        (REPO_ROOT / "reproducibility/models.json").read_text(encoding="utf-8")
    )
    errors: list[str] = []

    actual_python = platform.python_version()
    if actual_python != environment["python"]:
        fail(
            f"Python {actual_python}; expected {environment['python']}",
            errors,
        )
    for package, expected in environment["packages"].items():
        try:
            actual = metadata.version(package)
        except metadata.PackageNotFoundError:
            fail(f"missing Python package: {package}=={expected}", errors)
            continue
        if actual != expected:
            fail(f"{package}=={actual}; expected {expected}", errors)

    cuszp_root = Path(
        os.environ.get("CUSZP_ROOT", environment["cuszp"]["default_root"])
    ).expanduser()
    library = cuszp_root / "install" / "lib" / "libcuSZp.so"
    if not library.is_file():
        fail(f"missing cuSZp library: {library}", errors)
    actual_commit = command_output(["git", "-C", str(cuszp_root), "rev-parse", "HEAD"])
    expected_commit = environment["cuszp"]["commit"]
    if actual_commit != expected_commit:
        fail(
            f"cuSZp commit {actual_commit!r}; expected {expected_commit}",
            errors,
        )

    if args.scope == "full":
        cache_root = Path(
            os.environ.get(
                "HF_HUB_CACHE",
                Path(models["cache_root"]).expanduser(),
            )
        )
        for item in models["models"]:
            model_root = cache_root / item["cache_dir"]
            revision = item["revision"]
            snapshot = model_root / "snapshots" / revision
            ref = model_root / "refs" / "main"
            if not snapshot.is_dir():
                fail(f"missing model snapshot: {item['id']}@{revision}", errors)
            actual_ref = (
                ref.read_text(encoding="utf-8").strip() if ref.is_file() else None
            )
            if actual_ref != revision:
                fail(
                    f"{item['id']} refs/main={actual_ref!r}; expected {revision}",
                    errors,
                )

        if not args.allow_different_hardware:
            gpu = command_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total",
                    "--format=csv,noheader,nounits",
                ]
            )
            expected_gpu = environment["reviewed_hardware"]
            if gpu is None:
                fail("nvidia-smi is unavailable", errors)
            else:
                fields = [field.strip() for field in gpu.splitlines()[0].split(",")]
                if fields[0] != expected_gpu["gpu"]:
                    fail(f"GPU {fields[0]}; expected {expected_gpu['gpu']}", errors)
                if fields[1] != expected_gpu["driver"]:
                    fail(
                        f"driver {fields[1]}; expected {expected_gpu['driver']}",
                        errors,
                    )

    if args.check_results:
        manifest = REPO_ROOT / "reproducibility/formal_data.sha256"
        if not manifest.is_file():
            fail(
                "missing reproducibility/formal_data.sha256; run "
                "reproducibility/build_hash_manifest.py --write",
                errors,
            )
        else:
            verify_hash_manifest(manifest, errors)

    if errors:
        print(f"Environment verification failed with {len(errors)} issue(s).")
        return 1
    print(f"Environment verification passed ({args.scope} scope).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
