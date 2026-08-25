# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical errors for the Slurm launcher boundary."""

from __future__ import annotations


class SlurmLauncherError(RuntimeError):
    """Base error for structured Slurm launcher operations."""


class SlurmCommandError(SlurmLauncherError):
    """A Slurm command could not be executed successfully."""


class SlurmParseError(SlurmLauncherError, ValueError):
    """Slurm returned output that violates the requested format."""


class BatchRenderError(SlurmLauncherError, ValueError):
    """A resolved plan cannot be rendered as a safe batch script."""
