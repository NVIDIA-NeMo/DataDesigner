# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted retry submission lifecycle."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import PositiveInt, field_validator, model_validator

from data_designer.slurm.contracts import ArtifactReference, Identifier
from data_designer.slurm.state.base import StateRecord, validate_utc_timestamp


class RetryState(str, Enum):
    """Durable state of one failed-shard retry request."""

    PREPARED = "prepared"
    SUBMITTED = "submitted"
    COMPLETED = "completed"
    FAILED = "failed"


class RetryStatus(StateRecord):
    """Atomically replaced submission progress for one retry plan."""

    retry_id: Identifier
    run_id: Identifier
    retry_plan: ArtifactReference
    revision: PositiveInt
    updated_at: datetime
    state: RetryState
    array_job_id: PositiveInt | None = None

    _updated_at_is_utc = field_validator("updated_at")(validate_utc_timestamp)

    @model_validator(mode="after")
    def validate_scheduler_identity(self) -> RetryStatus:
        if self.state in {RetryState.SUBMITTED, RetryState.COMPLETED} and self.array_job_id is None:
            raise ValueError("submitted retry state requires an array job identity")
        if self.state is RetryState.PREPARED and self.array_job_id is not None:
            raise ValueError("prepared retry state cannot contain an array job identity")
        return self


def validate_retry_status_transition(previous: RetryStatus, current: RetryStatus) -> RetryStatus:
    """Require immutable retry identity and one-way submission progress."""
    if previous.retry_id != current.retry_id or previous.run_id != current.run_id:
        raise ValueError("retry status identity cannot change")
    if previous.retry_plan != current.retry_plan:
        raise ValueError("retry status plan identity cannot change")
    if current.revision != previous.revision + 1:
        raise ValueError("retry status revision must increase by one")
    if current.updated_at < previous.updated_at:
        raise ValueError("retry status timestamp cannot move backward")
    allowed = {
        RetryState.PREPARED: {RetryState.SUBMITTED, RetryState.FAILED},
        RetryState.SUBMITTED: {RetryState.COMPLETED, RetryState.FAILED},
        RetryState.COMPLETED: set(),
        RetryState.FAILED: set(),
    }
    if current.state not in allowed[previous.state]:
        raise ValueError(f"retry status cannot move from {previous.state.value!r} to {current.state.value!r}")
    return current


__all__ = ["RetryState", "RetryStatus", "validate_retry_status_transition"]
