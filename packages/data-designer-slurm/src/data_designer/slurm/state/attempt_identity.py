# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local binding of a retry task to its persisted attempt."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import AttemptId, Identifier, ShardId, validate_absolute_path
from data_designer.slurm.state.base import SchedulerIdentity
from data_designer.slurm.state.errors import SlurmStateError, StateConflictError
from data_designer.slurm.state.execution import AttemptLifecycleState
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_SHARD_ID_ADAPTER = TypeAdapter(ShardId)
_ATTEMPT_ID_ADAPTER = TypeAdapter(AttemptId)


def require_attempt_scheduler_identity(
    workspace_root: str | Path,
    run_id: Identifier,
    shard_id: ShardId,
    attempt_id: AttemptId,
    scheduler: SchedulerIdentity,
) -> None:
    """Require the complete persisted attempt chain to name this allocation."""
    root, normalized_run_id, normalized_shard_id, normalized_attempt_id = _validate_identity(
        workspace_root,
        run_id,
        shard_id,
        attempt_id,
    )
    storage = StateStorage(root, normalized_run_id)
    reader = StateReader(storage, normalized_run_id)
    run, plan, shard = reader.load_shard_context(normalized_shard_id)
    attempts = reader.load_validated_shard_attempts(run, plan, shard)
    attempt = reader.get_attempt(attempts, normalized_attempt_id)
    if attempt.scheduler != scheduler:
        raise StateConflictError("retry allocation does not match the persisted attempt scheduler identity")
    if attempt.state is not AttemptLifecycleState.SUBMITTED:
        raise StateConflictError("retry allocation requires an unstarted persisted attempt")


def _validate_identity(
    workspace_root: str | Path,
    run_id: Identifier,
    shard_id: ShardId,
    attempt_id: AttemptId,
) -> tuple[Path, Identifier, ShardId, AttemptId]:
    try:
        root = validate_absolute_path(Path(workspace_root).as_posix())
        normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
        normalized_shard_id = _SHARD_ID_ADAPTER.validate_python(shard_id, strict=True)
        normalized_attempt_id = _ATTEMPT_ID_ADAPTER.validate_python(attempt_id, strict=True)
    except (ValidationError, ValueError) as error:
        raise SlurmStateError("invalid retry allocation identity") from error
    return Path(root), normalized_run_id, normalized_shard_id, normalized_attempt_id


def main(argv: Sequence[str] | None = None) -> int:
    """Validate one allocation identity from explicit scheduler arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace-root", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--shard-id", required=True)
    parser.add_argument("--attempt-id", required=True)
    parser.add_argument("--array-job-id", required=True, type=int)
    parser.add_argument("--array-task-id", required=True, type=int)
    arguments = parser.parse_args(argv)
    require_attempt_scheduler_identity(
        arguments.workspace_root,
        arguments.run_id,
        arguments.shard_id,
        arguments.attempt_id,
        SchedulerIdentity(array_job_id=arguments.array_job_id, array_task_id=arguments.array_task_id),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main", "require_attempt_scheduler_identity"]
