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
from data_designer.slurm.planning import ResolvedSlurmRunPlan
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
from data_designer.slurm.state.destinations import CollectionDestinationResolver
from data_designer.slurm.state.outputs import RetryPlan

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
    ) -> SlurmRetryExecution:
        """Render or submit one sparse retry from persisted run state."""
        operation = SlurmServiceOperation.RETRY_RUN
        run_id = self._resolve_run_reference(run_or_job_id, operation)
        try:
            observed_at = self._clock()
            coordinator = SlurmRetryCoordinator(self._workspace_root, run_id, self._launcher)
            resolved_plan = SlurmStateWriter(self._workspace_root, run_id).load_resolved_plan()
            effective_resume_mode = self._resolve_retry_resume_mode(resolved_plan, resume)
            preview: tuple[RetryPlan, str] | None = None
            if effective_resume_mode is None:
                preview = coordinator.preview_active(shard_ids=shard_ids, observed_at=observed_at)
                if preview is None:
                    preview = coordinator.preview(
                        shard_ids=shard_ids,
                        effective_resume_mode="never",
                        observed_at=observed_at,
                    )
                    effective_resume_mode = self._resolve_if_possible_resume_mode(resolved_plan, preview[0], operation)
                else:
                    effective_resume_mode = preview[0].effective_resume_mode
                shard_ids = tuple(shard.shard_id for shard in preview[0].planned_shards)
            if dry_run:
                if preview is None or preview[0].effective_resume_mode != effective_resume_mode:
                    preview = coordinator.preview(
                        shard_ids=shard_ids,
                        effective_resume_mode=effective_resume_mode,
                        observed_at=observed_at,
                    )
                assert preview is not None
                plan, batch_script = preview
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
                        observed_at=observed_at,
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
        num_partitions: int | None,
    ) -> SlurmCollectionExecution:
        """Submit or recover one winner-driven collection."""
        operation = SlurmServiceOperation.COLLECT_RUN
        run_id = self._resolve_run_input_path(input_path, operation)
        try:
            plan = SlurmStateWriter(self._workspace_root, run_id).load_resolved_plan()
            try:
                CollectionDestinationResolver().resolve(plan, destination)
            except StateConflictError as error:
                raise SlurmServiceError(SlurmServiceErrorCode.INVALID_REQUEST, operation, str(error)) from None
            if num_partitions is not None and num_partitions != plan.output.partitions:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.INVALID_REQUEST,
                    operation,
                    "num_partitions must match the persisted run output partitions",
                )
            effective_partitions = plan.output.partitions
            status = SlurmCollectionCoordinator(self._workspace_root, run_id, self._launcher).submit(
                destination=destination,
                submitted_at=self._clock(),
            )
            if status.scheduler is None:
                raise SlurmServiceError(
                    SlurmServiceErrorCode.UNAVAILABLE,
                    operation,
                    "Slurm collection submission was not accepted",
                )
            return SlurmCollectionExecution(
                run_id=run_id,
                collection_id=status.collection_id,
                state=status.state,
                job_id=status.scheduler,
                output_path=destination.as_posix(),
                num_partitions=effective_partitions,
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
            raise SlurmServiceError(SlurmServiceErrorCode.CONFLICT, operation, str(error)) from None
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

    @staticmethod
    def _resolve_retry_resume_mode(
        plan: ResolvedSlurmRunPlan,
        requested: Literal["never", "always", "if_possible"],
    ) -> Literal["never", "always"] | None:
        if requested != "if_possible":
            return requested
        pinned = plan.invocation.authored.resume
        return None if pinned == "if_possible" else pinned

    @staticmethod
    def _resolve_if_possible_resume_mode(
        plan: ResolvedSlurmRunPlan,
        retry_plan: RetryPlan,
        operation: SlurmServiceOperation,
    ) -> Literal["never", "always"]:
        workspaces = {shard.shard_id: Path(shard.resume_workspace.path) for shard in plan.shards}
        availability = {_has_resume_data(workspaces[shard.shard_id]) for shard in retry_plan.planned_shards}
        if len(availability) != 1:
            raise SlurmServiceError(
                SlurmServiceErrorCode.INVALID_REQUEST,
                operation,
                "retry selection mixes resumable and fresh shards; choose --resume never or --resume always",
            )
        return "always" if availability.pop() else "never"

    def _resolve_run_reference(
        self,
        run_or_job_id: Identifier,
        operation: SlurmServiceOperation,
    ) -> Identifier:
        workspace_root = Path(self._workspace_root)
        direct = workspace_root / "runs" / run_or_job_id
        direct_match = run_or_job_id if direct.is_dir() else None
        if direct_match is not None and not run_or_job_id.isdecimal():
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
        targets = set(matches)
        if direct_match is not None:
            targets.add(direct_match)
        if not targets:
            raise SlurmServiceError(SlurmServiceErrorCode.NOT_FOUND, operation, "managed Slurm job was not found")
        if len(targets) != 1:
            raise SlurmServiceError(
                SlurmServiceErrorCode.CONFLICT,
                operation,
                "numeric reference matches multiple managed runs",
            )
        return targets.pop()

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


def _has_resume_data(path: Path) -> bool:
    return not path.is_symlink() and path.is_dir() and next(path.iterdir(), None) is not None


__all__ = ["RunRetryCollectionBackend"]
