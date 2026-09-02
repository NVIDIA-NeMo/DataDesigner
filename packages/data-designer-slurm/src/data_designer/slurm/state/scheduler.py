# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import field_validator, model_validator

from data_designer.slurm.state.base import (
    SchedulerJobIdentity,
    StateRecord,
    validate_optional_utc_timestamp,
    validate_utc_timestamp,
)


class SchedulerState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    NODE_FAILED = "node_failed"
    PREEMPTED = "preempted"
    REQUEUED = "requeued"
    OUT_OF_MEMORY = "out_of_memory"
    ACCOUNTING_LAG = "accounting_lag"
    UNKNOWN = "unknown"


class SchedulerObservation(StateRecord):
    """Normalized scheduler observation used for deterministic reconciliation."""

    scheduler: SchedulerJobIdentity
    observed_at: datetime
    state: SchedulerState
    reconciliation_deadline: datetime | None = None

    _observed_at_is_utc = field_validator("observed_at")(validate_utc_timestamp)
    _reconciliation_deadline_is_utc = field_validator("reconciliation_deadline")(validate_optional_utc_timestamp)

    @model_validator(mode="after")
    def validate_reconciliation_deadline(self) -> SchedulerObservation:
        if self.state is SchedulerState.ACCOUNTING_LAG:
            if self.reconciliation_deadline is None:
                raise ValueError("accounting lag requires a reconciliation deadline")
            if self.reconciliation_deadline < self.observed_at:
                raise ValueError("reconciliation deadline must not precede the observation")
        elif self.reconciliation_deadline is not None:
            raise ValueError("only accounting lag may have a reconciliation deadline")
        return self


class EffectiveAttemptState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ACCOUNTING_LAG = "accounting_lag"
    UNKNOWN = "unknown"


def is_scheduler_failure_state(state: SchedulerState) -> bool:
    """Return whether a scheduler state is terminal failure evidence."""
    return state in {
        SchedulerState.FAILED,
        SchedulerState.CANCELLED,
        SchedulerState.TIMED_OUT,
        SchedulerState.NODE_FAILED,
        SchedulerState.PREEMPTED,
        SchedulerState.REQUEUED,
        SchedulerState.OUT_OF_MEMORY,
    }


def is_scheduler_terminal_state(state: SchedulerState) -> bool:
    """Return whether a scheduler state is terminal accounting evidence."""
    return state is SchedulerState.COMPLETED or is_scheduler_failure_state(state)
