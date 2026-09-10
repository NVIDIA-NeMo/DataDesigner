# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public run-planning facade for Data Designer Slurm."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import Identifier, ShardId
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)
from data_designer.slurm.services.results import (
    SlurmCollectionExecution,
    SlurmPersistedRunStatus,
    SlurmRetryExecution,
    SlurmRunCancellation,
    SlurmRunExecution,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_SHARD_IDS_ADAPTER = TypeAdapter(tuple[ShardId, ...])


class SlurmRunPlanner(Protocol):
    """Resolve authored run configs through a supported service dependency."""

    def plan(self, config: DataDesignerSlurmConfig, *, source_root: Path) -> ResolvedSlurmRunPlan:
        """Return the immutable plan for one authored run.

        Any non-``INTERNAL`` service error must contain a caller-safe message.
        """


class SlurmBatchScriptRenderer(Protocol):
    """Render immutable plans through a supported service dependency."""

    def __call__(self, plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
        """Return the deterministic batch script for one attempt.

        Any non-``INTERNAL`` service error must contain a caller-safe message.
        """


class SlurmRunBackend(Protocol):
    """Execute and inspect runs through a supported service dependency."""

    def execute(
        self,
        config: DataDesignerSlurmConfig,
        *,
        source_root: Path,
        dry_run: bool,
        force: bool,
    ) -> SlurmRunExecution:
        """Render or submit one run."""

    def status(self, run_id: Identifier) -> SlurmPersistedRunStatus:
        """Return the durable M2 status for one run."""

    def cancel(self, run_id: Identifier) -> SlurmRunCancellation:
        """Request cancellation of active jobs."""

    def retry(
        self,
        run_or_job_id: Identifier,
        *,
        shard_ids: tuple[ShardId, ...] | None,
        resume: Literal["never", "always", "if_possible"],
        dry_run: bool,
        force: bool,
    ) -> SlurmRetryExecution:
        """Render or submit one sparse retry."""

    def collect(
        self,
        input_path: Path,
        *,
        destination: Path,
        num_partitions: int,
    ) -> SlurmCollectionExecution:
        """Submit or recover one winner-driven collection."""


class SlurmRunService:
    """Coordinate public run operations through package-owned boundaries.

    The service borrows its injected dependencies and does not manage their
    lifecycle.

    Args:
        planner: Run-planning dependency implementing ``SlurmRunPlanner``.
        renderer: Batch-rendering dependency implementing ``SlurmBatchScriptRenderer``.
        backend: Optional run-operation dependency. Package-owned factories provide it.
    """

    def __init__(
        self,
        planner: SlurmRunPlanner,
        renderer: SlurmBatchScriptRenderer,
        backend: SlurmRunBackend | None = None,
    ) -> None:
        self._planner = planner
        self._renderer = renderer
        self._backend = backend

    def plan(
        self,
        config: DataDesignerSlurmConfig,
        *,
        source_root: str | Path = ".",
    ) -> ResolvedSlurmRunPlan:
        """Resolve and compile one run without rendering or submission.

        The returned plan must reference the exact serialized authored config.

        Raises:
            SlurmServiceError: If the request is invalid or planning fails.
        """
        operation = SlurmServiceOperation.PLAN_RUN
        if not isinstance(config, DataDesignerSlurmConfig):
            raise _make_invalid_request_error(operation, "config must be a DataDesignerSlurmConfig")
        if not isinstance(source_root, str | Path):
            raise _make_invalid_request_error(operation, "source_root must be a path")

        def build_plan() -> ResolvedSlurmRunPlan:
            plan = self._planner.plan(config, source_root=Path(source_root).expanduser().resolve())
            if not isinstance(plan, ResolvedSlurmRunPlan):
                raise TypeError("run planner returned an invalid result")
            if plan.authored_config.sha256 != config.compute_sha256():
                raise ValueError("run plan does not match the requested authored config")
            return plan

        return _invoke_service_backend(operation, build_plan)

    def render_attempt(
        self,
        resolved_plan: ResolvedSlurmRunPlan,
        *,
        attempt_ordinal: int = 1,
    ) -> str:
        """Render one attempt from an already resolved immutable plan.

        Rendering never re-resolves the authored config or submits the script.

        Raises:
            SlurmServiceError: If the request is invalid or rendering fails.
        """
        operation = SlurmServiceOperation.RENDER_ATTEMPT
        if not isinstance(resolved_plan, ResolvedSlurmRunPlan):
            raise _make_invalid_request_error(operation, "resolved_plan must be a ResolvedSlurmRunPlan")
        if type(attempt_ordinal) is not int or attempt_ordinal <= 0:
            raise _make_invalid_request_error(operation, "attempt_ordinal must be a positive integer")

        def render() -> str:
            script = self._renderer(resolved_plan, attempt_ordinal=attempt_ordinal)
            if type(script) is not str or not script:
                raise TypeError("batch renderer returned an invalid script")
            return script

        return _invoke_service_backend(operation, render)

    def execute(
        self,
        config: DataDesignerSlurmConfig,
        *,
        source_root: str | Path = ".",
        dry_run: bool = False,
        force: bool = False,
    ) -> SlurmRunExecution:
        """Render or submit one run through package-owned production wiring."""
        operation = SlurmServiceOperation.EXECUTE_RUN
        if not isinstance(config, DataDesignerSlurmConfig):
            raise _make_invalid_request_error(operation, "config must be a DataDesignerSlurmConfig")
        if not isinstance(source_root, str | Path):
            raise _make_invalid_request_error(operation, "source_root must be a path")
        if type(dry_run) is not bool or type(force) is not bool:
            raise _make_invalid_request_error(operation, "dry_run and force must be booleans")
        backend = self._require_backend(operation)

        def execute_run() -> SlurmRunExecution:
            result = backend.execute(
                config,
                source_root=Path(source_root).expanduser().resolve(),
                dry_run=dry_run,
                force=force,
            )
            if not isinstance(result, SlurmRunExecution):
                raise TypeError("run backend returned an invalid execution result")
            return result

        return _invoke_service_backend(operation, execute_run)

    def status(self, run_id: Identifier) -> SlurmPersistedRunStatus:
        """Reconcile scheduler observations and return persisted M2 records."""
        operation = SlurmServiceOperation.STATUS_RUN
        normalized_run_id = _validate_run_id(run_id, operation)
        backend = self._require_backend(operation)

        def load_status() -> SlurmPersistedRunStatus:
            result = backend.status(normalized_run_id)
            if not isinstance(result, SlurmPersistedRunStatus) or result.run.run_id != normalized_run_id:
                raise TypeError("run backend returned an invalid status result")
            return result

        return _invoke_service_backend(operation, load_status)

    def cancel(self, run_id: Identifier) -> SlurmRunCancellation:
        """Request cancellation of active jobs; reconciliation updates persisted state."""
        operation = SlurmServiceOperation.CANCEL_RUN
        normalized_run_id = _validate_run_id(run_id, operation)
        backend = self._require_backend(operation)

        def cancel_run() -> SlurmRunCancellation:
            result = backend.cancel(normalized_run_id)
            if not isinstance(result, SlurmRunCancellation) or result.run_id != normalized_run_id:
                raise TypeError("run backend returned an invalid cancellation result")
            return result

        return _invoke_service_backend(operation, cancel_run)

    def retry(
        self,
        run_or_job_id: Identifier,
        *,
        shard_ids: Sequence[ShardId] | None = None,
        resume: Literal["never", "always", "if_possible"] = "if_possible",
        dry_run: bool = False,
        force: bool = False,
    ) -> SlurmRetryExecution:
        """Render or submit retry attempts for failed shards."""
        operation = SlurmServiceOperation.RETRY_RUN
        normalized_reference = _validate_run_id(run_or_job_id, operation)
        normalized_shards = _validate_shard_ids(shard_ids, operation)
        if resume not in {"never", "always", "if_possible"}:
            raise _make_invalid_request_error(operation, "resume must be 'never', 'always', or 'if_possible'")
        if type(dry_run) is not bool or type(force) is not bool:
            raise _make_invalid_request_error(operation, "dry_run and force must be booleans")
        backend = self._require_backend(operation)

        def retry_run() -> SlurmRetryExecution:
            result = backend.retry(
                normalized_reference,
                shard_ids=normalized_shards,
                resume=resume,
                dry_run=dry_run,
                force=force,
            )
            if not isinstance(result, SlurmRetryExecution):
                raise TypeError("run backend returned an invalid retry result")
            if normalized_shards is not None and result.shard_ids != normalized_shards:
                raise TypeError("run backend returned retry shards that do not match the request")
            if result.state != ("dry_run" if dry_run else "submitted"):
                raise TypeError("run backend returned a retry state that does not match the request")
            return result

        return _invoke_service_backend(operation, retry_run)

    def collect(
        self,
        input_path: str | Path,
        *,
        destination: str | Path,
        num_partitions: int = 1,
    ) -> SlurmCollectionExecution:
        """Submit or recover collection for one managed run directory."""
        operation = SlurmServiceOperation.COLLECT_RUN
        if not isinstance(input_path, str | Path) or not isinstance(destination, str | Path):
            raise _make_invalid_request_error(operation, "input_path and destination must be paths")
        if type(num_partitions) is not int or num_partitions <= 0:
            raise _make_invalid_request_error(operation, "num_partitions must be a positive integer")
        normalized_input = Path(input_path).expanduser().resolve()
        normalized_destination = Path(destination).expanduser().resolve()
        backend = self._require_backend(operation)

        def collect_run() -> SlurmCollectionExecution:
            result = backend.collect(
                normalized_input,
                destination=normalized_destination,
                num_partitions=num_partitions,
            )
            if not isinstance(result, SlurmCollectionExecution):
                raise TypeError("run backend returned an invalid collection result")
            if result.output_path != normalized_destination.as_posix() or result.num_partitions != num_partitions:
                raise TypeError("run backend returned collection intent that does not match the request")
            return result

        return _invoke_service_backend(operation, collect_run)

    def _require_backend(self, operation: SlurmServiceOperation) -> SlurmRunBackend:
        if self._backend is None:
            raise SlurmServiceError(
                SlurmServiceErrorCode.UNAVAILABLE,
                operation,
                "run operations require package-owned service construction",
            )
        return self._backend


def _validate_run_id(run_id: object, operation: SlurmServiceOperation) -> Identifier:
    try:
        return _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
    except ValidationError:
        raise _make_invalid_request_error(operation, "run_id must be a valid identifier") from None


def _validate_shard_ids(
    shard_ids: Sequence[ShardId] | None,
    operation: SlurmServiceOperation,
) -> tuple[ShardId, ...] | None:
    if shard_ids is None:
        return None
    if isinstance(shard_ids, str | bytes):
        raise _make_invalid_request_error(operation, "shard_ids must be a sequence of shard identifiers")
    try:
        normalized = _SHARD_IDS_ADAPTER.validate_python(tuple(shard_ids), strict=True)
    except (TypeError, ValidationError):
        raise _make_invalid_request_error(operation, "shard_ids must contain valid shard identifiers") from None
    if not normalized or len(normalized) != len(set(normalized)):
        raise _make_invalid_request_error(operation, "shard_ids must be non-empty and unique when provided")
    return tuple(sorted(normalized))
