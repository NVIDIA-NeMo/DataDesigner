# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the Slurm tools needed inside a generated allocation."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from data_designer.slurm.contracts import validate_absolute_path


class SchedulerPathError(ValueError):
    """The submit environment cannot provide the required allocation tools."""


def resolve_scheduler_bin_path(configured: str | None) -> str:
    """Find a single executable directory for the batch job's controlled PATH."""
    if configured is None:
        srun = shutil.which("srun")
        if srun is None:
            raise SchedulerPathError(
                "srun was not found on the submit host PATH; load your site's Slurm environment before submitting"
            )
        directory = Path(os.path.abspath(srun)).parent
    else:
        directory = Path(configured)
    try:
        validate_absolute_path(str(directory))
    except ValueError:
        raise SchedulerPathError("scheduler.bin_path must be one absolute directory") from None
    if ":" in str(directory):
        raise SchedulerPathError("scheduler.bin_path must be one absolute directory")
    if not directory.is_dir() or any(
        not (directory / executable).is_file() or not os.access(directory / executable, os.X_OK)
        for executable in ("srun", "scontrol")
    ):
        raise SchedulerPathError(
            "the Slurm executable directory must contain runnable srun and scontrol; "
            "load your site's Slurm environment or set scheduler.bin_path to their shared directory"
        )
    return str(directory)
