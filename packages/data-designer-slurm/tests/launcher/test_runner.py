# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from collections.abc import Mapping, Sequence

import pytest

from data_designer.slurm.launcher import SubprocessRunner


def test_subprocess_runner_uses_argv_and_only_explicit_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    observed: dict[str, object] = {}

    def fake_run(
        command: Sequence[str],
        *,
        check: bool,
        stdin: int,
        capture_output: bool,
        text: bool,
        encoding: str,
        errors: str,
        env: Mapping[str, str],
        timeout: float,
    ) -> subprocess.CompletedProcess[str]:
        observed.update(
            command=command,
            check=check,
            stdin=stdin,
            capture_output=capture_output,
            text=text,
            encoding=encoding,
            errors=errors,
            env=env,
            timeout=timeout,
        )
        return subprocess.CompletedProcess(command, 0, stdout="ok\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    supplied_environment = {"PATH": "/usr/bin", "LC_ALL": "fr_FR.UTF-8"}
    runner = SubprocessRunner(environment=supplied_environment, timeout_seconds=4.0)
    supplied_environment["SECRET"] = "must-not-leak"

    completed = runner.run(("squeue", "--noheader"))

    assert completed.stdout == "ok\n"
    assert observed == {
        "command": ("squeue", "--noheader"),
        "check": False,
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "text": True,
        "encoding": "utf-8",
        "errors": "replace",
        "env": {"LC_ALL": "C", "PATH": "/usr/bin"},
        "timeout": 4.0,
    }


def test_subprocess_runner_default_environment_forwards_only_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "/workspace/slurm/bin:/usr/bin")
    monkeypatch.setenv("SECRET", "must-not-leak")

    runner = SubprocessRunner()

    assert runner.environment == {"LC_ALL": "C", "PATH": "/workspace/slurm/bin:/usr/bin"}


def test_subprocess_runner_default_environment_replaces_empty_search_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PATH", "")

    runner = SubprocessRunner()

    assert runner.environment == {"LC_ALL": "C", "PATH": os.defpath}


def test_subprocess_runner_environment_is_immutable() -> None:
    runner = SubprocessRunner()

    with pytest.raises(TypeError):
        runner.environment["SECRET"] = "value"  # type: ignore[index]


@pytest.mark.parametrize("timeout_seconds", [0, -1, True, "30", float("nan"), float("inf")])
def test_subprocess_runner_rejects_invalid_timeout(timeout_seconds: object) -> None:
    with pytest.raises(ValueError, match="finite positive"):
        SubprocessRunner(timeout_seconds=timeout_seconds)  # type: ignore[arg-type]


@pytest.mark.parametrize("environment", ({"BAD=NAME": "value"}, {"NAME": "bad\0value"}))
def test_subprocess_runner_rejects_invalid_environment(environment: dict[str, str]) -> None:
    with pytest.raises(ValueError, match="environment"):
        SubprocessRunner(environment=environment)
