# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execution-scoped runtime log paths and restrictive file creation."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path

from data_designer.slurm.runtime.models import RuntimeStep
from data_designer.slurm.state.filesystem import (
    ensure_private_child_directory,
    open_verified_child_directory,
    open_verified_directory,
)

_LOG_DIRECTORY_NAME = "logs"
_EXECUTION_DIRECTORY_PREFIX = "execution-"


def execution_log_directory(attempt_directory: Path, readiness_revision: int) -> Path:
    """Return the package-owned log directory for one execution epoch."""
    if type(readiness_revision) is not int or readiness_revision < 1:
        raise ValueError("execution readiness revision must be positive")
    return attempt_directory / _LOG_DIRECTORY_NAME / f"{_EXECUTION_DIRECTORY_PREFIX}{readiness_revision:08d}"


def bind_execution_logs(step: RuntimeStep, log_directory: Path) -> RuntimeStep:
    """Bind one immutable step specification to its execution-scoped logs."""
    return replace(
        step,
        stdout_path=log_directory / f"{step.step_id}.out",
        stderr_path=log_directory / f"{step.step_id}.err",
    )


@contextmanager
def open_step_logs(step: RuntimeStep) -> Iterator[tuple[int, int]]:
    """Create and open one step's restrictive logs below verified directories."""
    log_directory = step.stdout_path.parent
    logs_root = log_directory.parent
    attempt_directory = logs_root.parent
    if logs_root.name != _LOG_DIRECTORY_NAME or not log_directory.name.startswith(_EXECUTION_DIRECTORY_PREFIX):
        raise OSError("runtime logs must use a package-owned execution directory")

    with open_verified_directory(attempt_directory, require_private=True) as attempt_descriptor:
        ensure_private_child_directory(attempt_descriptor, _LOG_DIRECTORY_NAME, logs_root)
    with open_verified_directory(logs_root, require_private=True) as logs_descriptor:
        ensure_private_child_directory(logs_descriptor, log_directory.name, log_directory)
        with open_verified_child_directory(logs_descriptor, log_directory.name, log_directory) as directory_descriptor:
            with ExitStack() as resources:
                stdout_descriptor = _create_log_file(directory_descriptor, step.stdout_path.name)
                resources.callback(os.close, stdout_descriptor)
                stderr_descriptor = _create_log_file(directory_descriptor, step.stderr_path.name)
                resources.callback(os.close, stderr_descriptor)
                yield stdout_descriptor, stderr_descriptor


def _create_log_file(directory_descriptor: int, name: str) -> int:
    return os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
        0o600,
        dir_fd=directory_descriptor,
    )


__all__ = ["bind_execution_logs", "execution_log_directory", "open_step_logs"]
