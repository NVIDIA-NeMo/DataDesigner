# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Annotated, Any, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field


class WorkflowStageMetadata(BaseModel):
    """Common metadata for a composite workflow stage."""

    model_config = ConfigDict(extra="allow")

    status: Literal["running", "failed", "completed", "completed_empty", "skipped_empty_upstream"]
    index: int
    name: str
    stage_dir: str
    depends_on: list[str]
    allow_empty: bool
    on_success_version: str | None
    output_processors: list[dict[str, Any]]
    output: str
    sampling_strategy: str
    selection_strategy: dict[str, Any] | None


class _StartedWorkflowStageMetadata(WorkflowStageMetadata):
    fingerprint: str
    num_records_requested: int
    seeded_from_stage: str | None
    seed_path: str | None
    config: dict[str, Any]


class RunningWorkflowStageMetadata(_StartedWorkflowStageMetadata):
    """Metadata for a running workflow stage."""

    status: Literal["running"]


class FailedWorkflowStageMetadata(_StartedWorkflowStageMetadata):
    """Metadata for a failed workflow stage."""

    status: Literal["failed"]
    duration_sec: float | None = None


class CompletedWorkflowStageMetadata(_StartedWorkflowStageMetadata):
    """Metadata for a completed workflow stage."""

    status: Literal["completed", "completed_empty"]
    num_records_actual: int
    output_records: int
    output_seed_path: str
    callback_output_path: str | None
    stage_output_override_path: str | None = None
    output_processor_output_path: str | None
    duration_sec: float


class SkippedWorkflowStageMetadata(WorkflowStageMetadata):
    """Metadata for a stage skipped after an empty upstream stage."""

    status: Literal["skipped_empty_upstream"]
    upstream_stage: str


_WorkflowStageMetadataVariant: TypeAlias = Annotated[
    RunningWorkflowStageMetadata
    | FailedWorkflowStageMetadata
    | CompletedWorkflowStageMetadata
    | SkippedWorkflowStageMetadata,
    Field(discriminator="status"),
]


class WorkflowMetadata(BaseModel):
    """Metadata persisted for a composite workflow run."""

    model_config = ConfigDict(extra="allow")

    name: str
    library_version: str
    stages: list[_WorkflowStageMetadataVariant]
