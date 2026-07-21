#!/usr/bin/env python3
"""Build or verify the SHA-256 manifest for canonical measured artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "reproducibility" / "formal_artifacts.json"
MANIFEST_PATH = REPO_ROOT / "reproducibility" / "formal_data.sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def artifact_paths() -> list[Path]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    suffixes = set(config["included_suffixes"])
    selected: set[Path] = set()

    for relative in config["files"]:
        path = REPO_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing canonical artifact: {relative}")
        selected.add(path)

    for relative in config["roots"]:
        root = REPO_ROOT / relative
        if not root.is_dir():
            raise FileNotFoundError(f"missing canonical directory: {relative}")
        selected.update(
            path for path in root.rglob("*")
            if path.is_file() and path.suffix in suffixes
        )

    return sorted(selected, key=lambda path: path.relative_to(REPO_ROOT).as_posix())


def render_manifest() -> str:
    return "".join(
        f"{digest(path)}  {path.relative_to(REPO_ROOT).as_posix()}\n"
        for path in artifact_paths()
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Replace formal_data.sha256 with hashes of the current artifacts.",
    )
    args = parser.parse_args()
    rendered = render_manifest()

    if args.write:
        MANIFEST_PATH.write_text(rendered, encoding="utf-8")
        print(f"Wrote {MANIFEST_PATH.relative_to(REPO_ROOT)}")
        return 0

    if not MANIFEST_PATH.is_file():
        print(f"Missing {MANIFEST_PATH.relative_to(REPO_ROOT)}", file=sys.stderr)
        return 1
    if MANIFEST_PATH.read_text(encoding="utf-8") != rendered:
        print("Canonical artifact manifest does not match the files.", file=sys.stderr)
        return 1
    print("Canonical artifact manifest matches.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
