# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persisted collection lifecycle and output records."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Annotated

from pydantic import Field, NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm.contracts import ArtifactReference, Identifier, Sha256Digest, validate_relative_path
from data_designer.slurm.state.base import (
    SchedulerJobIdentity,
    StateRecord,
    StateValue,
    validate_optional_utc_timestamp,
    validate_utc_timestamp,
)
from data_designer.slurm.state.scheduler import SchedulerObservation


class CollectionState(str, Enum):
    """Persisted lifecycle for one CPU collection job."""

    PREPARED = "prepared"
    SUBMITTED = "submitted"
    PENDING = "pending"
    RUNNING = "running"
    ACCOUNTING_LAG = "accounting_lag"
    UNKNOWN = "unknown"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class CollectedOutputFile(StateValue):
    """One deterministic file in a published collected dataset."""

    relative_path: str
    sha256: Sha256Digest
    byte_size: NonNegativeInt
    record_count: NonNegativeInt
    modified_at_ns: NonNegativeInt
    changed_at_ns: NonNegativeInt

    _relative_path_is_safe = field_validator("relative_path")(validate_relative_path)


class CollectionResult(StateRecord):
    """Immutable proof of a completely staged collection output."""

    collection_id: Identifier
    run_id: Identifier
    completed_at: datetime
    collection_plan_sha256: Sha256Digest
    actual_records: NonNegativeInt
    files: tuple[CollectedOutputFile, ...] = Field(min_length=1)

    _completed_at_is_utc = field_validator("completed_at")(validate_utc_timestamp)

    @model_validator(mode="after")
    def validate_files(self) -> CollectionResult:
        paths = tuple(output.relative_path for output in self.files)
        if len(paths) != len(set(paths)):
            raise ValueError("collected output paths must be unique")
        if sum(output.record_count for output in self.files) != self.actual_records:
            raise ValueError("collected output row counts must equal actual_records")
        return self


class CollectionStatus(StateRecord):
    """Atomically replaced scheduler and publication state for one collection."""

    collection_id: Identifier
    run_id: Identifier
    collection_plan: ArtifactReference
    staging_directory: Annotated[
        str,
        StringConstraints(pattern=r"^\.dd-collection-[0-9a-f]{32}\.tmp$"),
    ]
    revision: PositiveInt
    updated_at: datetime
    state: CollectionState
    scheduler: SchedulerJobIdentity | None = None
    scheduler_observation: SchedulerObservation | None = None
    result: ArtifactReference | None = None
    reconciliation_deadline: datetime | None = None

    _updated_at_is_utc = field_validator("updated_at")(validate_utc_timestamp)
    _reconciliation_deadline_is_utc = field_validator("reconciliation_deadline")(validate_optional_utc_timestamp)

    @model_validator(mode="after")
    def validate_evidence(self) -> CollectionStatus:
        if self.state is CollectionState.PREPARED:
            if self.scheduler is not None or self.scheduler_observation is not None or self.result is not None:
                raise ValueError("prepared collection cannot contain scheduler or result evidence")
            if self.reconciliation_deadline is None or self.reconciliation_deadline <= self.updated_at:
                raise ValueError("prepared collection requires a future reconciliation deadline")
            return self
        if self.reconciliation_deadline is not None:
            raise ValueError("settled collection cannot contain a reconciliation deadline")
        if self.state is not CollectionState.FAILED and self.scheduler is None:
            raise ValueError("submitted collection states require a scheduler identity")
        if self.scheduler is None and self.scheduler_observation is not None:
            raise ValueError("collection observation requires a scheduler identity")
        if self.scheduler_observation is not None and self.scheduler_observation.scheduler != self.scheduler:
            raise ValueError("collection scheduler observation identity does not match")
        if (self.state is CollectionState.SUCCEEDED) != (self.result is not None):
            raise ValueError("collection result is required exactly for succeeded state")
        return self
