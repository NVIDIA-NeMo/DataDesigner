# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.contracts import (
    ArtifactReference,
    AttemptId,
    Identifier,
    RecordRange,
    ResumeWorkspace,
    ShardId,
)
from data_designer.slurm.state.base import (
    SchedulerIdentity,
    StateRecord,
    validate_utc_timestamp,
)


class RunManifest(StateRecord):
    """Identity and immutable authored/resolved inputs for a Slurm run."""

    run_id: Identifier
    created_at: datetime
    authored_config: ArtifactReference
    resolved_plan: ArtifactReference
    shard_count: PositiveInt

    _created_at_is_utc = field_validator("created_at")(validate_utc_timestamp)


class ShardManifest(StateRecord):
    """Stable shard identity and planner-owned input partition reference."""

    run_id: Identifier
    shard_id: ShardId
    shard_index: NonNegativeInt
    record_range: RecordRange
    input_partition: ArtifactReference | None = None
    resume_workspace: ResumeWorkspace
    created_at: datetime

    _created_at_is_utc = field_validator("created_at")(validate_utc_timestamp)


class AttemptLifecycleState(str, Enum):
    CREATED = "created"
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class AttemptTerminalClassification(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NODE_FAILED = "node_failed"
    PREEMPTED = "preempted"
    REQUEUED = "requeued"
    OUT_OF_MEMORY = "out_of_memory"
    UNKNOWN = "unknown"


class AttemptManifest(StateRecord):
    """Attempt identity, lifecycle, scheduler identity, and output reference."""

    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    attempt_ordinal: PositiveInt
    resolved_plan: ArtifactReference
    state: AttemptLifecycleState
    terminal_classification: AttemptTerminalClassification | None = None
    scheduler: SchedulerIdentity | None = None
    candidate_output: ArtifactReference | None = None
    created_at: datetime
    updated_at: datetime

    _timestamps_are_utc = field_validator("created_at", "updated_at")(validate_utc_timestamp)

    @model_validator(mode="after")
    def validate_lifecycle(self) -> AttemptManifest:
        if self.updated_at < self.created_at:
            raise ValueError("updated_at must not precede created_at")

        if self.state is not AttemptLifecycleState.CREATED and self.scheduler is None:
            raise ValueError("submitted and later attempts require scheduler identity")

        terminal = self.state in {
            AttemptLifecycleState.SUCCEEDED,
            AttemptLifecycleState.FAILED,
        }
        if terminal != (self.terminal_classification is not None):
            raise ValueError("terminal classification must be present exactly for terminal attempts")

        if self.state is AttemptLifecycleState.SUCCEEDED:
            if self.terminal_classification is not AttemptTerminalClassification.SUCCEEDED:
                raise ValueError("successful attempts require a succeeded terminal classification")
            if self.candidate_output is None:
                raise ValueError("successful attempts require a candidate output reference")
        elif self.terminal_classification is AttemptTerminalClassification.SUCCEEDED:
            raise ValueError("failed attempts cannot have a succeeded terminal classification")

        return self
