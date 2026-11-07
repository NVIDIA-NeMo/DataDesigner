# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized Slurm planning errors."""

from __future__ import annotations


class SlurmPlanningError(ValueError):
    """Base error for Slurm configuration resolution and plan validation."""


class SlurmConfigResolutionError(SlurmPlanningError):
    """Raised when resolved inputs do not match one authored declaration."""


class SlurmPlanCompilationError(SlurmPlanningError):
    """Raised when one effective configuration cannot produce a valid plan."""


class SlurmPlanContractError(SlurmPlanningError):
    """Raised when a resolved plan does not match its authored inputs."""
