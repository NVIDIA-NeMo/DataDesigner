# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Structured Slurm submission, observation, and batch rendering."""

from __future__ import annotations

from data_designer.slurm.launcher.client import SlurmCommandClient, SlurmExecutables
from data_designer.slurm.launcher.errors import (
    BatchRenderError,
    SlurmCommandError,
    SlurmLauncherError,
    SlurmParseError,
)
from data_designer.slurm.launcher.models import (
    AccountingRecord,
    QueueRecord,
    SlurmExitCode,
    SlurmSubmission,
)
from data_designer.slurm.launcher.renderer import BatchDirective, render_batch_script
from data_designer.slurm.launcher.runner import CommandRunner, SubprocessRunner

__all__ = [
    "AccountingRecord",
    "BatchDirective",
    "BatchRenderError",
    "CommandRunner",
    "QueueRecord",
    "SlurmCommandClient",
    "SlurmCommandError",
    "SlurmExecutables",
    "SlurmExitCode",
    "SlurmLauncherError",
    "SlurmParseError",
    "SlurmSubmission",
    "SubprocessRunner",
    "render_batch_script",
]
