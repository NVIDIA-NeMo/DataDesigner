# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small shared utilities for authored Slurm configuration."""

from __future__ import annotations


def convert_duration_to_seconds(value: str) -> int:
    """Convert a validated Slurm duration to whole seconds."""
    factor = {"s": 1, "m": 60, "h": 3600, "d": 86400}[value[-1]]
    return int(value[:-1]) * factor
