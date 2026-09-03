# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal normalized errors for the Slurm launcher boundary."""

from __future__ import annotations


class SlurmLauncherError(RuntimeError):
    """Base error for structured Slurm launcher operations."""


class SlurmCommandError(SlurmLauncherError):
    """A Slurm command could not be executed successfully."""

    def __init__(self, message: str, *, command_may_have_completed: bool = False) -> None:
        super().__init__(message)
        self.command_may_have_completed = command_may_have_completed


class SlurmSubmissionError(SlurmLauncherError):
    """An sbatch submission failed with an explicit ambiguity classification."""

    def __init__(self, message: str, *, may_have_succeeded: bool) -> None:
        super().__init__(message)
        self.may_have_succeeded = may_have_succeeded


class SlurmCommandOutputError(SlurmLauncherError, ValueError):
    """A Slurm command returned output that violates its requested format."""


class SlurmBatchRenderError(SlurmLauncherError, ValueError):
    """A resolved plan cannot be rendered as a safe batch script."""
