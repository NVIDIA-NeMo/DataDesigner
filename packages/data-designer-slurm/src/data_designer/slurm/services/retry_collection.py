# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Production retry and collection adapters for the public run service."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import Identifier, ShardId
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmLauncherError
from data_designer.slurm.services.errors import SlurmServiceError, SlurmServiceErrorCode, SlurmServiceOperation
from data_designer.slurm.services.results import SlurmCollectionExecution, SlurmRetryExecution
from data_designer.slurm.state import (
    SlurmCollectionCoordinator,
    SlurmRetryCoordinator,
    SlurmStateError,
    SlurmStateWriter,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class RunRetryCollectionBackend:
    """Adapt public run operations to persisted retry and collection capabilities."""

    def __init__(self, workspace_root: str, launcher: SlurmCommandClient, clock: Callable[[], datetime]) -> None:
        self._workspace_root = workspace_root
        self._launcher = launcher
        self._clock = clock

    def retry(
        self,
        run_or_job_id: Identifier,
        *,
        shard_ids: tuple[ShardId, ...] | None,
        resume: Literal["never", "always", "if_possible"],
        dry_run: bool,
        force: bool,
    ) -> SlurmRetryExecution:
        """Render or submit one sparse retry from persisted run state."""
        del force
        operation = SlurmServiceOperation.RETRY_RUN
        run_id = self._resolve_run_reference(run_or_job_id, operation)
        try:
            effective_resume_mode = self._resolve_retry_resume_mode(run_id, resume)
            coordinator = SlurmRetryCoordinator(self._workspace_root, run_id, self._launcher)
            if dry_run:
                plan, batch_script = coordinator.preview(
                    shard_ids=shard_ids,
                    effective_resume_mode=effective_resume_mode,
                    observed_at=self._clock(),
                )
                return SlurmRetryExecution(
                    run_id=run_id,
                    state="dry_run",
                    shard_ids=tuple(shard.shard_id for shard in plan.planned_shards),
                    attempt_ids=tuple(shard.attempt_id for shard in plan.planned_shards),
                    effective_resume_mode=effective_resume_mode,
                    batch_script=batch_script,
                )
            attempts = tuple(
                sorted(
                    coordinator.retry(
                        shard_ids=shard_ids,
                        effective_resume_mode=effective_resume_mode,
                        observed_at=self._clock(),
                    ),
                    key=lambda attempt: attempt.shard_id,
                )
            )
            scheduler_ids = {attempt.scheduler.array_job_id for attempt in attempts if attempt.scheduler is not None}
            if not attempts or len(scheduler_ids) != 1 or any(attempt.scheduler is None for attempt in attempts):
                raise SlurmStateError("retry did not return one accepted sparse array")
            return SlurmRetryExecution(
                run_id=run_id,
                state="submitted",
                shard_ids=tuple(attempt.shard_id for attempt in attempts),
                attempt_ids=tuple(attempt.attempt_id for attempt in attempts),
                effective_resume_mode=effective_resume_mode,
                job_id=scheduler_ids.pop(),
            )
        except StateNotFoundError:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "run state was not found") from None
        except StateConflictError as error:
            raise SlurmServiceError(SlurmServiceErrorCode.CONFLICT, operation, str(error)) from None
        except StateCorruptionError:
            raise SlurmServiceError(SlurmServiceErrorCode.INTERNAL, operation, "retry state is inconsistent") from None
        except SlurmStateError as error:
            if _has_cause(error, SlurmLauncherError):
                raise SlurmServiceError(
                    SlurmServiceErrorCode.UNAVAILABLE,
                    operation,
                    "Slurm retry submission is unavailable",
                ) from None
            raise SlurmServiceError(SlurmServiceErrorCode.INTERNAL, operation, "retry state cannot be read") from None

    def collect(
        self,
        input_path: Path,
        *,
        destination: Path,
        num_partitions: int,
    ) -> SlurmCollectionExecution:
        """Submit or recover one winner-driven collection."""
        operation = SlurmServiceOperation.COLLECT_RUN
        run_id = self._resolve_run_input_path(input_path, operation)
        try:
            plan = SlurmStateWriter(self._workspace_root, run_id).load_resolved_plan()
            if num_partitions != plan.output.partitions:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    operation,
                    "num_partitions must match the persisted run output partitions",
                )
            status = SlurmCollectionCoordinator(self._workspace_root, run_id, self._launcher).submit(
                destination=destination,
                submitted_at=self._clock(),
            )
            if status.scheduler is None:
                raise SlurmStateError("collection did not return an accepted Slurm job")
            return SlurmCollectionExecution(
                run_id=run_id,
                collection_id=status.collection_id,
                state=status.state,
                job_id=status.scheduler,
                output_path=destination.as_posix(),
                num_partitions=num_partitions,
            )
        except SlurmServiceError:
            raise
        except StateNotFoundError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.NOT_FOUND,
                operation,
                "run state or shard winner was not found",
            ) from None
        except StateConflictError as error:
            code = (
                SlurmServiceErrorCode.INVALID_REQUEST
                if str(error).startswith("collection destination")
                else SlurmServiceErrorCode.CONFLICT
            )
            raise SlurmServiceError(code, operation, str(error)) from None
        except StateCorruptionError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                operation,
                "collection state is inconsistent",
            ) from None
        except SlurmStateError as error:
            if _has_cause(error, SlurmLauncherError):
                raise SlurmServiceError(
                    SlurmServiceErrorCode.UNAVAILABLE,
                    operation,
                    "Slurm collection submission is unavailable",
                ) from None
            raise SlurmServiceError(
                SlurmServiceErrorCode.INTERNAL,
                operation,
                "collection state cannot be read",
            ) from None

    def _resolve_retry_resume_mode(
        self,
        run_id: Identifier,
        requested: Literal["never", "always", "if_possible"],
    ) -> Literal["never", "always"]:
        if requested != "if_possible":
            return requested
        pinned = SlurmStateWriter(self._workspace_root, run_id).load_resolved_plan().invocation.authored.resume
        return "always" if pinned == "if_possible" else pinned

    def _resolve_run_reference(
        self,
        run_or_job_id: Identifier,
        operation: SlurmServiceOperation,
    ) -> Identifier:
        workspace_root = Path(self._workspace_root)
        direct = workspace_root / "runs" / run_or_job_id
        if direct.is_dir():
            return run_or_job_id
        if not run_or_job_id.isdecimal() or int(run_or_job_id) <= 0:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "run state was not found")
        job_id = int(run_or_job_id)
        matches: list[Identifier] = []
        runs_root = workspace_root / "runs"
        if runs_root.is_dir():
            for candidate in sorted(runs_root.iterdir(), key=lambda path: path.name):
                if not candidate.is_dir():
                    continue
                try:
                    candidate_id = _IDENTIFIER_ADAPTER.validate_python(candidate.name, strict=True)
                    writer = SlurmStateWriter(workspace_root, candidate_id)
                    attempts = tuple(
                        attempt for shard in writer.load_shards() for attempt in writer.load_attempts(shard.shard_id)
                    )
                except (ValidationError, StateNotFoundError):
                    continue
                except SlurmStateError:
                    raise SlurmServiceError(
                        SlurmServiceErrorCode.INTERNAL,
                        operation,
                        "managed run state cannot be searched safely",
                    ) from None
                if any(
                    attempt.scheduler is not None and attempt.scheduler.array_job_id == job_id for attempt in attempts
                ):
                    matches.append(candidate_id)
        if not matches:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "managed Slurm job was not found")
        if len(matches) != 1:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                operation,
                "Slurm job ID matches multiple managed runs",
            )
        return matches[0]

    def _resolve_run_input_path(
        self,
        input_path: Path,
        operation: SlurmServiceOperation,
    ) -> Identifier:
        runs_root = (Path(self._workspace_root) / "runs").resolve()
        if input_path.parent != runs_root:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "input_path must identify a managed run directory",
            )
        try:
            run_id = _IDENTIFIER_ADAPTER.validate_python(input_path.name, strict=True)
        except ValidationError:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "input_path must contain a valid managed run ID",
            ) from None
        if not input_path.is_dir():
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "run state was not found")
        return run_id


def _has_cause(error: BaseException, expected_type: type[BaseException]) -> bool:
    cause = error.__cause__
    while cause is not None:
        if isinstance(cause, expected_type):
            return True
        cause = cause.__cause__
    return False


__all__ = ["RunRetryCollectionBackend"]
