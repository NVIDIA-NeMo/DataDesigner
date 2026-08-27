# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Emit factual image metadata using only the target image's Python standard library."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
from collections.abc import Iterable
from pathlib import Path

INSPECTOR_VERSION = "inspector-1"
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_CLIENT_DISTRIBUTIONS = (
    "data-designer",
    "data-designer-config",
    "data-designer-engine",
    "data-designer-slurm",
    "pip",
)


def inspect_image(kind: str, sqsh_sha256: str) -> dict[str, object]:
    """Return one JSON-compatible digest-bound inspection record."""
    if _SHA256_PATTERN.fullmatch(sqsh_sha256) is None:
        raise ValueError("SQSH digest must be lowercase SHA-256 text")
    if kind == "client":
        inspection = _inspect_client()
    elif kind == "serving":
        inspection = _inspect_serving()
    else:
        raise ValueError("image kind must be 'client' or 'serving'")
    return {
        "schema_version": 1,
        "inspector_version": INSPECTOR_VERSION,
        "sqsh_sha256": sqsh_sha256,
        "inspection": inspection,
    }


def write_inspection(path: Path, record: dict[str, object]) -> None:
    """Atomically write one restrictive inspection record."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def main(arguments: list[str] | None = None) -> int:
    """Run the standalone inspector entrypoint."""
    values = sys.argv[1:] if arguments is None else arguments
    if len(values) != 3:
        raise ValueError("expected KIND SQSH_SHA256 OUTPUT_PATH")
    kind, sqsh_sha256, output_path = values
    output = Path(output_path)
    if not output.is_absolute():
        raise ValueError("inspection output path must be absolute")
    write_inspection(output, inspect_image(kind, sqsh_sha256))
    return 0


def _inspect_client() -> dict[str, object]:
    versions = _list_distribution_versions(importlib.metadata.distributions())
    missing = tuple(name for name in _REQUIRED_CLIENT_DISTRIBUTIONS if name not in versions)
    if missing:
        raise RuntimeError(f"required client distributions are not installed: {', '.join(missing)}")
    installer_path = shutil.which("pip")
    if installer_path is None or not Path(installer_path).is_absolute():
        raise RuntimeError("required executable 'pip' is not installed at an absolute path")
    cache_tag = sys.implementation.cache_tag
    if cache_tag is None:
        raise RuntimeError("active Python does not expose an ABI cache tag")
    python_abi = f"cp{cache_tag.removeprefix('cpython-')}" if cache_tag.startswith("cpython-") else cache_tag
    return {
        "kind": "client",
        "python_implementation": platform.python_implementation().casefold(),
        "python_version": platform.python_version(),
        "python_abi": python_abi,
        "distributions": [{"name": name, "version": version} for name, version in sorted(versions.items())],
        "installer_path": Path(installer_path).as_posix(),
        "installer_version": versions["pip"],
    }


def _inspect_serving() -> dict[str, object]:
    executable_path = shutil.which("vllm")
    if executable_path is None or not Path(executable_path).is_absolute():
        raise RuntimeError("required executable 'vllm' is not installed at an absolute path")
    try:
        runtime_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError("required distribution 'vllm' is not installed") from error
    return {
        "kind": "serving",
        "server_type": "vllm",
        "runtime_version": runtime_version,
        "executable_path": Path(executable_path).as_posix(),
    }


def _list_distribution_versions(distributions: Iterable[importlib.metadata.Distribution]) -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in distributions:
        raw_name = distribution.metadata.get("Name")
        if not raw_name:
            raise RuntimeError("installed distribution is missing its canonical name")
        name = re.sub(r"[-_.]+", "-", raw_name).casefold()
        existing = versions.setdefault(name, distribution.version)
        if existing != distribution.version:
            raise RuntimeError(f"installed distribution {name!r} has conflicting versions")
    return versions


if __name__ == "__main__":
    raise SystemExit(main())
