# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical errors for Slurm image inspection and registry operations."""

from __future__ import annotations


class SlurmImageError(RuntimeError):
    """Base error for package-owned Slurm image operations."""


class ImageInspectionError(SlurmImageError):
    """Raised when an image environment cannot produce valid factual metadata."""


class ImageRegistryError(SlurmImageError):
    """Raised when persisted image registry state cannot be read or written."""


class ImageNotFoundError(SlurmImageError):
    """Raised when an image alias or path is not registered."""


class ImageConflictError(SlurmImageError):
    """Raised when an image registration conflicts with immutable registry state."""


class ImageVerificationError(SlurmImageError):
    """Raised when an image file or inspection record fails verification."""


class ImageLifecycleError(SlurmImageError):
    """Raised when a CPU Slurm image lifecycle job cannot be prepared or submitted."""
