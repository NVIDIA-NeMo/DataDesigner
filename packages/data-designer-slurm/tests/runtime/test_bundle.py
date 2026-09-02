# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import os
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

import data_designer.slurm.runtime.bundle as runtime_bundle
from data_designer.slurm.runtime.bundle import stage_runtime_bundle
from data_designer.slurm.runtime.errors import SlurmRuntimeError


def test_runtime_bundle_is_deterministic_content_addressed_and_restrictive(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)

    first = stage_runtime_bundle(workspace)
    second = stage_runtime_bundle(workspace)

    assert first == second
    archive_path = Path(first.path)
    content = archive_path.read_bytes()
    assert hashlib.sha256(content).hexdigest() == first.sha256
    assert archive_path.name == f"{first.sha256}.tar.gz"
    assert stat.S_IMODE(archive_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(archive_path.parent.stat().st_mode) == 0o700
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:gz") as archive:
        names = archive.getnames()
        assert names[0] == "entrypoint.sh"
        assert names[1] == "data_designer/slurm/__init__.py"
        assert "data_designer/slurm/runtime/controller.py" in names
        assert "data_designer/slurm/runtime/entrypoint.py" in names
        assert archive.getmember("entrypoint.sh").mode == 0o500
        assert all(archive.getmember(name).uid == 0 for name in names)
        entrypoint = archive.extractfile("entrypoint.sh")
        assert entrypoint is not None
        assert b'PYTHONPATH="${runtime_root}"' in entrypoint.read()


def test_runtime_bundle_rejects_different_bytes_at_digest_path(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    reference = stage_runtime_bundle(workspace)
    path = Path(reference.path)
    path.write_bytes(b"different")
    path.chmod(0o600)

    with pytest.raises(SlurmRuntimeError, match="cannot stage"):
        stage_runtime_bundle(workspace)


def test_runtime_bundle_rejects_symlinked_runtime_root(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir(mode=0o700)
    outside.mkdir(mode=0o700)
    (workspace / "runtime").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SlurmRuntimeError, match="cannot stage"):
        stage_runtime_bundle(workspace)

    assert tuple(outside.iterdir()) == ()


def test_runtime_bundle_rejects_a_group_writable_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o770)
    workspace.chmod(0o770)

    with pytest.raises(SlurmRuntimeError, match="cannot stage"):
        stage_runtime_bundle(workspace)


def test_runtime_bundle_normalizes_source_read_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)

    def fail_archive() -> bytes:
        raise OSError("injected source race")

    monkeypatch.setattr(runtime_bundle, "_build_runtime_archive", fail_archive)

    with pytest.raises(SlurmRuntimeError, match="cannot stage"):
        stage_runtime_bundle(workspace)


def test_extracted_bundle_runtime_takes_precedence_over_installed_sources(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    extracted = tmp_path / "extracted"
    workspace.mkdir(mode=0o700)
    extracted.mkdir(mode=0o700)
    reference = stage_runtime_bundle(workspace)
    with tarfile.open(reference.path, mode="r:gz") as archive:
        archive.extractall(extracted, filter="data")
    environment = dict(os.environ)
    environment["PYTHONPATH"] = extracted.as_posix()

    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            "import data_designer.slurm.runtime as runtime; print(runtime.__file__, end='')",
        ),
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == (extracted / "data_designer/slurm/runtime/__init__.py").as_posix()
