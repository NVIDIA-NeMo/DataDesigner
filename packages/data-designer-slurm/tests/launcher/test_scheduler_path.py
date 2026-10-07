# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from pathlib import Path

import pytest

from data_designer.slurm.launcher.scheduler_path import SchedulerPathError, resolve_scheduler_bin_path


def _install_fake_slurm_tools(directory: Path) -> None:
    directory.mkdir()
    for name in ("srun", "scontrol"):
        executable = directory / name
        executable.write_text("#!/bin/sh\nexit 0\n")
        executable.chmod(0o755)


def test_scheduler_directory_is_discovered_from_submit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "slurm-bin"
    _install_fake_slurm_tools(directory)
    monkeypatch.setenv("PATH", str(directory))

    assert resolve_scheduler_bin_path(None) == str(directory)


def test_explicit_override_works_without_submit_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "slurm-bin"
    _install_fake_slurm_tools(directory)
    monkeypatch.setenv("PATH", os.defpath)

    assert resolve_scheduler_bin_path(str(directory)) == str(directory)


def test_missing_srun_reports_actionable_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", str(tmp_path))

    with pytest.raises(SchedulerPathError, match="srun was not found"):
        resolve_scheduler_bin_path(None)


def test_missing_scontrol_fails_before_submission(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    directory = tmp_path / "slurm-bin"
    _install_fake_slurm_tools(directory)
    (directory / "scontrol").unlink()
    monkeypatch.setenv("PATH", str(directory))

    with pytest.raises(SchedulerPathError, match="runnable srun and scontrol"):
        resolve_scheduler_bin_path(None)
