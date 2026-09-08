# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public run-planning facade for Data Designer Slurm."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import Identifier
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import (
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)
from data_designer.slurm.services.results import (
    SlurmPersistedRunStatus,
    SlurmRunCancellation,
    SlurmRunExecution,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


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
        """Cancel active managed jobs recorded for one run."""


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
        """Return persisted M2 records without scheduler reconciliation."""
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
        """Cancel active managed jobs identified by persisted M2 records."""
        operation = SlurmServiceOperation.CANCEL_RUN
        normalized_run_id = _validate_run_id(run_id, operation)
        backend = self._require_backend(operation)

        def cancel_run() -> SlurmRunCancellation:
            result = backend.cancel(normalized_run_id)
            if not isinstance(result, SlurmRunCancellation) or result.run_id != normalized_run_id:
                raise TypeError("run backend returned an invalid cancellation result")
            return result

        return _invoke_service_backend(operation, cancel_run)

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
