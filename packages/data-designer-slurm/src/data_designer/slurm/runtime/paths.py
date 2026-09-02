# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Translate reviewed host paths through resolved container mounts."""

from __future__ import annotations

import posixpath

from data_designer.slurm.contracts import validate_absolute_path
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode


def get_container_path(
    plan: ResolvedSlurmRunPlan,
    host_path: str,
    *,
    require_writable: bool = False,
) -> str:
    """Map one absolute host path through the most specific resolved mount."""
    try:
        normalized = validate_absolute_path(host_path)
    except ValueError as error:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime host path is invalid") from error
    candidates = tuple(
        mount
        for mount in plan.container_mounts
        if normalized == mount.source or normalized.startswith(f"{mount.source}/")
    )
    if not candidates:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "runtime path is not available through a resolved container mount",
        )
    mount = max(candidates, key=lambda value: len(value.source))
    if require_writable and mount.read_only:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "runtime path requires a writable resolved container mount",
        )
    relative_path = posixpath.relpath(normalized, mount.source)
    mapped = mount.target if relative_path == "." else posixpath.join(mount.target, relative_path)
    return validate_absolute_path(mapped)


__all__ = ["get_container_path"]
