# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public run-planning facade for Data Designer Slurm."""

from __future__ import annotations

from typing import Protocol

from pydantic import PositiveInt, field_validator, model_validator

from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import ContractValue
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.services.errors import (
    SlurmServiceOperation,
    invalid_request,
    invoke_backend,
)


class RunPlanningBackend(Protocol):
    """Resolve and compile one authored run without exposing implementation types."""

    def plan(self, config: DataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return the immutable plan for one authored run."""


class BatchScriptRenderer(Protocol):
    """Render one immutable run plan as a thin batch entrypoint."""

    def __call__(self, plan: ResolvedSlurmRunPlan, *, attempt_ordinal: int = 1) -> str:
        """Return the deterministic batch script for one attempt."""


class SlurmRunPlanResult(ContractValue):
    """Correlated immutable plan and rendered M0 batch script."""

    plan: ResolvedSlurmRunPlan
    attempt_ordinal: PositiveInt
    batch_script: str

    @field_validator("batch_script")
    @classmethod
    def validate_batch_script(cls, value: str) -> str:
        if not value.startswith("#!/usr/bin/env bash\n"):
            raise ValueError("batch script must start with the package bash shebang")
        if any(
            character in "\u2028\u2029"
            or (character not in "\n\t" and (ord(character) < 32 or 127 <= ord(character) <= 159))
            for character in value
        ):
            raise ValueError("batch script must not contain control or non-LF line-separator characters")
        return value

    @model_validator(mode="after")
    def validate_plan_binding(self) -> SlurmRunPlanResult:
        lines = self.batch_script.split("\n")
        expected_digest = f'readonly DD_PLAN_SHA256="{self.plan.compute_sha256()}"'
        digest_declarations = [line for line in lines if line.lstrip(" \t").startswith("readonly DD_PLAN_SHA256")]
        if digest_declarations != [expected_digest]:
            raise ValueError("batch script must bind the exact resolved plan digest")
        expected_attempt = f'readonly DD_ATTEMPT_ORDINAL="{self.attempt_ordinal:04d}"'
        attempt_declarations = [line for line in lines if line.lstrip(" \t").startswith("readonly DD_ATTEMPT_ORDINAL")]
        if attempt_declarations != [expected_attempt]:
            raise ValueError("batch script must bind the requested attempt ordinal")
        return self


class SlurmRunService:
    """Coordinate public run operations through injected package boundaries."""

    def __init__(self, planner: RunPlanningBackend, renderer: BatchScriptRenderer) -> None:
        self._planner = planner
        self._renderer = renderer

    def plan(
        self,
        config: DataDesignerSlurmConfig,
        *,
        attempt_ordinal: int = 1,
    ) -> SlurmRunPlanResult:
        """Resolve, compile, and render one run without submission or persistence."""
        operation = SlurmServiceOperation.PLAN_RUN
        if not isinstance(config, DataDesignerSlurmConfig):
            raise invalid_request(operation, "config must be a DataDesignerSlurmConfig")
        if type(attempt_ordinal) is not int or attempt_ordinal <= 0:
            raise invalid_request(operation, "attempt_ordinal must be a positive integer")

        def build_result() -> SlurmRunPlanResult:
            plan = self._planner.plan(config)
            if not isinstance(plan, ResolvedSlurmRunPlan):
                raise TypeError("run planner returned an invalid result")
            if plan.authored_config.sha256 != config.compute_sha256():
                raise ValueError("run plan does not match the requested authored config")
            return SlurmRunPlanResult(
                plan=plan,
                attempt_ordinal=attempt_ordinal,
                batch_script=self._renderer(plan, attempt_ordinal=attempt_ordinal),
            )

        return invoke_backend(operation, build_result)
