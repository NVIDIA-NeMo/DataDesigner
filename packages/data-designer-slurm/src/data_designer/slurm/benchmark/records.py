# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated

from pydantic import (
    Field,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

from data_designer.slurm._contracts import ContractRecord, ContractValue, Identifier
from data_designer.slurm.planning import ArtifactReference


class BenchmarkChildRun(ContractValue):
    case_id: Identifier
    child_run_id: Identifier
    child_config: ArtifactReference


class BenchmarkManifest(ContractRecord):
    """Stable mapping from benchmark cases to ordinary child runs."""

    benchmark_id: Identifier
    benchmark_config: ArtifactReference
    children: tuple[BenchmarkChildRun, ...] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_children(self) -> BenchmarkManifest:
        case_ids = tuple(child.case_id for child in self.children)
        run_ids = tuple(child.child_run_id for child in self.children)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark case IDs must be unique")
        if len(run_ids) != len(set(run_ids)):
            raise ValueError("benchmark child run IDs must be unique")
        return self


class BenchmarkOutcome(str, Enum):
    PENDING = "pending"
    ACCOUNTING_LAG = "accounting_lag"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    INCOMPLETE = "incomplete"


class BenchmarkCaseResult(ContractValue):
    case_id: Identifier
    child_run_id: Identifier
    outcome: BenchmarkOutcome
    topology_digest: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
    requested_records: PositiveInt
    actual_records: NonNegativeInt | None = None
    boot_seconds: NonNegativeFloat | None = None
    generation_seconds: NonNegativeFloat | None = None
    wall_seconds: NonNegativeFloat | None = None
    rows_per_second: NonNegativeFloat | None = None
    request_count: NonNegativeInt | None = None
    token_count: NonNegativeInt | None = None
    gpus_per_job: PositiveInt
    nodes_per_job: PositiveInt
    gpu_hours_per_job: NonNegativeFloat | None = None
    total_gpu_hours: NonNegativeFloat | None = None
    target_jobs: PositiveInt | None = None
    feasible: bool | None = None

    @model_validator(mode="after")
    def validate_metrics(self) -> BenchmarkCaseResult:
        if self.actual_records is not None and self.actual_records > self.requested_records:
            raise ValueError("benchmark actual_records must not exceed requested_records")
        required = (
            self.actual_records,
            self.boot_seconds,
            self.generation_seconds,
            self.wall_seconds,
            self.rows_per_second,
            self.gpu_hours_per_job,
            self.total_gpu_hours,
            self.target_jobs,
            self.feasible,
        )
        if self.outcome is BenchmarkOutcome.SUCCEEDED and any(value is None for value in required):
            raise ValueError("successful benchmark cases require complete timing and feasibility metrics")
        return self


class BenchmarkRecommendationKind(str, Enum):
    PARETO = "pareto"
    MINIMUM_JOBS = "minimum_jobs"
    MINIMUM_GPU_HOURS = "minimum_gpu_hours"


class BenchmarkRecommendation(ContractValue):
    kind: BenchmarkRecommendationKind
    case_id: Identifier


class BenchmarkReport(ContractRecord):
    """Atomic point-in-time benchmark analysis output."""

    benchmark_id: Identifier
    analysis_id: Identifier
    benchmark_manifest: ArtifactReference
    created_at: datetime
    cases: tuple[BenchmarkCaseResult, ...] = Field(min_length=1)
    recommendations: tuple[BenchmarkRecommendation, ...] = ()

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("created_at must be timezone-aware UTC")
        return value

    @model_validator(mode="after")
    def validate_report(self) -> BenchmarkReport:
        case_ids = tuple(case.case_id for case in self.cases)
        if len(case_ids) != len(set(case_ids)):
            raise ValueError("benchmark report case IDs must be unique")
        unknown = {recommendation.case_id for recommendation in self.recommendations}.difference(case_ids)
        if unknown:
            raise ValueError(f"recommendations reference unknown cases: {', '.join(sorted(unknown))}")
        kinds = tuple(recommendation.kind for recommendation in self.recommendations)
        if len(kinds) != len(set(kinds)):
            raise ValueError("benchmark recommendation kinds must be unique")
        return self
