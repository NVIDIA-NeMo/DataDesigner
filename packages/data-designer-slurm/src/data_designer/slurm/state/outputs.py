# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.state.base import (
    ArtifactReference,
    Identifier,
    Sha256Digest,
    StateRecord,
    StateValue,
    validate_absolute_path,
    validate_relative_path,
    validate_utc_timestamp,
)


class CandidateOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    EMPTY = "empty"


class CandidateOutputFile(StateValue):
    """One immutable file contained in a candidate output."""

    relative_path: str
    sha256: Sha256Digest
    byte_size: NonNegativeInt
    record_count: NonNegativeInt

    _relative_path_is_safe = field_validator("relative_path")(validate_relative_path)


class CandidateOutputManifest(StateRecord):
    """Attempt-local output that may become the immutable shard winner."""

    run_id: Identifier
    shard_id: Identifier
    attempt_id: Identifier
    attempt_ordinal: PositiveInt
    created_at: datetime
    dataset_path: str
    requested_records: PositiveInt
    actual_records: NonNegativeInt
    require_exact_record_count: bool
    outcome: CandidateOutcome
    files: tuple[CandidateOutputFile, ...]
    dataset_schema_digest: Sha256Digest
    provenance_digest: Sha256Digest

    _created_at_is_utc = field_validator("created_at")(validate_utc_timestamp)
    _dataset_path_is_safe = field_validator("dataset_path")(validate_absolute_path)

    @property
    def winner_eligible(self) -> bool:
        """Whether policy permits publishing this candidate as the shard winner."""
        return self.actual_records > 0 and (
            not self.require_exact_record_count or self.actual_records == self.requested_records
        )

    @model_validator(mode="after")
    def validate_output(self) -> CandidateOutputManifest:
        if self.actual_records > self.requested_records:
            raise ValueError("actual_records must not exceed requested_records")

        expected_outcome = (
            CandidateOutcome.EMPTY
            if self.actual_records == 0
            else CandidateOutcome.COMPLETE
            if self.actual_records == self.requested_records
            else CandidateOutcome.PARTIAL
        )
        if self.outcome is not expected_outcome:
            raise ValueError(f"outcome must be {expected_outcome.value!r} for this record count")

        relative_paths = [output_file.relative_path for output_file in self.files]
        if len(relative_paths) != len(set(relative_paths)):
            raise ValueError("candidate output file paths must be unique")
        if self.actual_records > 0 and not self.files:
            raise ValueError("non-empty candidate outputs require at least one file")
        if sum(output_file.record_count for output_file in self.files) != self.actual_records:
            raise ValueError("candidate output file record counts must equal actual_records")
        return self


class ShardWinner(StateRecord):
    """Immutable pointer selecting exactly one candidate for a shard."""

    run_id: Identifier
    shard_id: Identifier
    attempt_id: Identifier
    attempt_ordinal: PositiveInt
    candidate_manifest: ArtifactReference
    published_at: datetime

    _published_at_is_utc = field_validator("published_at")(validate_utc_timestamp)


class CollectionShard(StateValue):
    """Winner manifest selected for one shard in a collection plan."""

    shard_id: Identifier
    winner_manifest: ArtifactReference


class CollectionPlan(StateRecord):
    """Immutable inputs and destinations for deterministic collection."""

    collection_id: Identifier
    run_id: Identifier
    created_at: datetime
    resolved_plan: ArtifactReference
    planned_shards: tuple[CollectionShard, ...] = Field(min_length=1)
    host_destination: str
    container_destination: str
    num_partitions: PositiveInt
    overwrite: Literal[False] = False

    _created_at_is_utc = field_validator("created_at")(validate_utc_timestamp)
    _destinations_are_safe = field_validator("host_destination", "container_destination")(validate_absolute_path)

    @model_validator(mode="after")
    def validate_shards(self) -> CollectionPlan:
        shard_ids = [shard.shard_id for shard in self.planned_shards]
        winner_paths = [shard.winner_manifest.path for shard in self.planned_shards]
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("collection shard IDs must be unique")
        if len(winner_paths) != len(set(winner_paths)):
            raise ValueError("collection winner manifest paths must be unique")
        return self
