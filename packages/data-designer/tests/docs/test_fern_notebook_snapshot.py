# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tarfile
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[4] / "docs" / "scripts" / "fern_notebook_snapshot.sh"


def write_notebooks(root: Path) -> dict[str, str]:
    notebooks = {
        "example.json": '{"cells": [{"output": "rendered"}]}\n',
        "manifest.ts": 'export const notebooks = ["example"];\n',
    }
    notebook_dir = root / "fern" / "components" / "notebooks"
    notebook_dir.mkdir(parents=True)
    for name, content in notebooks.items():
        (notebook_dir / name).write_text(content)
    return notebooks


@pytest.mark.parametrize("mode", ["executed", "source-fallback"])
def test_snapshot_round_trip(tmp_path: Path, mode: str) -> None:
    root = tmp_path / "website"
    expected = write_notebooks(root)
    archive = tmp_path / f"notebooks-{mode}.tar.gz"
    env = {**os.environ, "GITHUB_RUN_ID": "123", "GITHUB_RUN_ATTEMPT": "2"}

    subprocess.run(
        [SCRIPT_PATH, "create", root, "v1.2.3", mode, archive],
        check=True,
        env=env,
    )

    metadata = json.loads((root / "fern" / "notebook-snapshot.json").read_text())
    digest = metadata.pop("sha256")
    assert metadata == {
        "schema_version": 1,
        "release_tag": "v1.2.3",
        "asset": archive.name,
        "mode": mode,
        "run_id": "123",
        "run_attempt": "2",
    }
    assert len(digest) == 64

    notebook_dir = root / "fern" / "components" / "notebooks"
    (notebook_dir / "example.json").write_text("stale")
    subprocess.run([SCRIPT_PATH, "restore", root, archive], check=True)

    assert {path.name: path.read_text() for path in notebook_dir.iterdir()} == expected


def test_restore_rejects_invalid_checksum_without_replacing_notebooks(tmp_path: Path) -> None:
    root = tmp_path / "website"
    write_notebooks(root)
    archive = tmp_path / "notebooks.tar.gz"
    subprocess.run([SCRIPT_PATH, "create", root, "v1.2.3", "executed", archive], check=True)
    archive.write_bytes(archive.read_bytes() + b"corrupt")

    result = subprocess.run([SCRIPT_PATH, "restore", root, archive], check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "checksum mismatch" in result.stderr
    assert (root / "fern" / "components" / "notebooks" / "example.json").exists()


def test_restore_rejects_asset_name_mismatch(tmp_path: Path) -> None:
    root = tmp_path / "website"
    write_notebooks(root)
    archive = tmp_path / "notebooks.tar.gz"
    subprocess.run([SCRIPT_PATH, "create", root, "v1.2.3", "executed", archive], check=True)
    renamed_archive = tmp_path / "different-name.tar.gz"
    renamed_archive.write_bytes(archive.read_bytes())

    result = subprocess.run(
        [SCRIPT_PATH, "restore", root, renamed_archive],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "asset mismatch" in result.stderr
    assert (root / "fern" / "components" / "notebooks" / "example.json").exists()


def test_restore_rejects_unexpected_archive_paths_without_replacing_notebooks(tmp_path: Path) -> None:
    root = tmp_path / "website"
    write_notebooks(root)
    archive = tmp_path / "notebooks.tar.gz"
    subprocess.run([SCRIPT_PATH, "create", root, "v1.2.3", "executed", archive], check=True)

    unexpected_file = tmp_path / "unexpected.txt"
    unexpected_file.write_text("unexpected")
    with tarfile.open(archive, "w:gz") as snapshot:
        snapshot.add(unexpected_file, arcname="unexpected.txt")

    metadata_path = root / "fern" / "notebook-snapshot.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["sha256"] = hashlib.sha256(archive.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))

    result = subprocess.run([SCRIPT_PATH, "restore", root, archive], check=False, capture_output=True, text=True)

    assert result.returncode != 0
    assert "unexpected paths" in result.stderr
    assert (root / "fern" / "components" / "notebooks" / "example.json").exists()
