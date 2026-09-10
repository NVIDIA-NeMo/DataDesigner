# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load allocation identity from container-visible persisted state."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import ValidationError

from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.state import SlurmStateWriter
from data_designer.slurm.state.filesystem import open_verified_directory, read_regular_text

_MAXIMUM_RECORD_SIZE = 16 * 1024 * 1024


def load_allocation_context(
    plan_path: Path,
    attempt_directory: Path,
    environment: Mapping[str, str],
) -> tuple[AllocationContext, SlurmStateWriter]:
    """Load one scheduler-selected shard attempt through its container paths."""
    writer = _load_state_writer(plan_path, attempt_directory)
    plan = writer.load_resolved_plan()
    expected_plan_path = Path(plan.authored_config.path).with_name("resolved-plan.json")
    if plan_path.as_posix() != get_container_path(plan, expected_plan_path.as_posix()):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "runtime plan path does not match persisted run intent",
        )
    shard = _select_shard(plan.shards, _scheduler_task_id(environment.get("SLURM_ARRAY_TASK_ID")))
    host_attempt_directory = expected_plan_path.parent / "shards" / shard.shard_id / "attempts" / attempt_directory.name
    if attempt_directory.as_posix() != get_container_path(
        plan, host_attempt_directory.as_posix(), require_writable=True
    ):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "attempt directory does not match the scheduler array task",
        )
    attempt = writer.load_attempt(shard.shard_id, attempt_directory.name)
    array_job_id = _scheduler_task_id(environment.get("SLURM_ARRAY_JOB_ID"))
    if attempt.scheduler is None or attempt.scheduler.array_job_id != array_job_id:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "scheduler array job does not match the persisted attempt",
        )
    return AllocationContext(plan, shard, attempt, host_attempt_directory), writer


def _load_state_writer(plan_path: Path, attempt_directory: Path) -> SlurmStateWriter:
    if not plan_path.is_absolute() or not attempt_directory.is_absolute():
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime paths must be absolute")
    if plan_path.name != "resolved-plan.json":
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "resolved plan path is invalid")
    try:
        with open_verified_directory(plan_path.parent, require_private=True) as descriptor:
            content = read_regular_text(
                descriptor,
                plan_path.name,
                plan_path,
                maximum_size=_MAXIMUM_RECORD_SIZE,
            )
        plan = ResolvedSlurmRunPlan.model_validate_json(content)
    except (OSError, UnicodeError, ValueError, ValidationError) as error:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT, "resolved plan is unavailable or invalid"
        ) from error
    run_id = plan.run_id
    logical_workspace_root = plan.selected_profile.profile.workspace_root
    logical_plan_path = Path(logical_workspace_root) / "runs" / run_id / plan_path.name
    if get_container_path(plan, logical_plan_path.as_posix()) != plan_path.as_posix():
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "resolved plan path is invalid")
    workspace_root = Path(get_container_path(plan, logical_workspace_root, require_writable=True))
    return SlurmStateWriter(
        workspace_root,
        run_id,
        logical_workspace_root=logical_workspace_root,
        local_path_resolver=lambda path: get_container_path(plan, path, require_writable=True),
    )


def _select_shard(shards: tuple[PlannedShard, ...], task_id: int) -> PlannedShard:
    selected = tuple(shard for shard in shards if shard.array_task_index == task_id)
    if len(selected) != 1:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "scheduler array task does not identify exactly one planned shard",
        )
    return selected[0]


def _scheduler_task_id(value: str | None) -> int:
    if value is None or not value.isascii() or not value.isdigit():
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "SLURM_ARRAY_TASK_ID must be a non-negative integer",
        )
    return int(value)


__all__ = ["load_allocation_context"]
