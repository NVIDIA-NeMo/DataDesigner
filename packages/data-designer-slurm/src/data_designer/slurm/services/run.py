# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public run-planning facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import (
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)


class _RunPlanner(Protocol):
    """Resolve one run; any normalized failure must be caller-safe."""

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return the immutable plan for one authored run."""


class _BatchScriptRenderer(Protocol):
    """Render one immutable plan; any normalized failure must be caller-safe."""

    def __call__(self, plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
        """Return the deterministic batch script for one attempt."""


class SlurmRunService:
    """Coordinate public run operations through package-owned boundaries.

    The service borrows its injected dependencies and does not manage their
    lifecycle.

    Args:
        planner: Package-owned run-planning boundary.
        renderer: Package-owned batch-rendering boundary.
    """

    def __init__(self, planner: _RunPlanner, renderer: _BatchScriptRenderer) -> None:
        self._planner = planner
        self._renderer = renderer

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Resolve and compile one run without rendering or submission.

        The returned plan must reference the exact serialized authored config.

        Raises:
            SlurmServiceError: If the request is invalid or planning fails.
        """
        operation = SlurmServiceOperation.PLAN_RUN
        if not isinstance(config, DataDesignerSlurmConfig):
            raise _make_invalid_request_error(operation, "config must be a DataDesignerSlurmConfig")

        def build_plan() -> ResolvedSlurmRunPlan:
            plan = self._planner.plan(config)
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
