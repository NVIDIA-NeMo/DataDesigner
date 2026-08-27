# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public run-planning facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from pydantic import PositiveInt

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import ContractValue
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import (
    SlurmServiceOperation,
    _invoke_service_backend,
    _make_invalid_request_error,
)


class _RunPlanner(Protocol):
    """Resolve and compile one authored run without exposing implementation types."""

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return the immutable plan for one authored run."""


class _BatchScriptRenderer(Protocol):
    """Render one immutable run plan as a thin batch entrypoint."""

    def __call__(self, plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int) -> str:
        """Return the deterministic batch script for one attempt."""


class RenderedSlurmAttempt(ContractValue):
    """In-process result for one rendered attempt of an immutable plan."""

    resolved_plan: ResolvedSlurmRunPlan
    attempt_ordinal: PositiveInt
    rendered_batch_script: str


class SlurmRunService:
    """Coordinate public run operations through injected package boundaries."""

    def __init__(self, planner: _RunPlanner, renderer: _BatchScriptRenderer) -> None:
        self._planner = planner
        self._renderer = renderer

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Resolve and compile one run without rendering or submission."""
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
    ) -> RenderedSlurmAttempt:
        """Render one attempt from an already resolved immutable plan."""
        operation = SlurmServiceOperation.RENDER_ATTEMPT
        if not isinstance(resolved_plan, ResolvedSlurmRunPlan):
            raise _make_invalid_request_error(operation, "resolved_plan must be a ResolvedSlurmRunPlan")
        if type(attempt_ordinal) is not int or attempt_ordinal <= 0:
            raise _make_invalid_request_error(operation, "attempt_ordinal must be a positive integer")

        def render() -> RenderedSlurmAttempt:
            return RenderedSlurmAttempt(
                resolved_plan=resolved_plan,
                attempt_ordinal=attempt_ordinal,
                rendered_batch_script=self._renderer(resolved_plan, attempt_ordinal=attempt_ordinal),
            )

        return _invoke_service_backend(operation, render)
