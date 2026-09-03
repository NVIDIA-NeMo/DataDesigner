# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Canonical errors for persisted Slurm run state."""

from __future__ import annotations


class SlurmStateError(RuntimeError):
    """Base error for state storage and publication."""


class StateNotFoundError(SlurmStateError):
    """Raised when requested persisted state does not exist."""


class StateConflictError(SlurmStateError):
    """Raised when a write conflicts with immutable or newer state."""


class StateCorruptionError(SlurmStateError):
    """Raised when persisted state cannot be safely read or validated."""
