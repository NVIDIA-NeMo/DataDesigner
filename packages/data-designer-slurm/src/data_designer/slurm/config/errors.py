# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Normalized authored Slurm configuration errors."""

from __future__ import annotations


class SlurmConfigError(ValueError):
    """Base error for authored Slurm configuration boundaries."""


class SlurmConfigBuilderError(SlurmConfigError):
    """Raised when the Slurm config builder is incomplete or cannot serialize."""


class SlurmConfigLoadError(SlurmConfigError):
    """Raised when a local Slurm configuration file is not strict and valid."""
