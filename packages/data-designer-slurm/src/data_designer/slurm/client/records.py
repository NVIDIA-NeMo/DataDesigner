# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm._contracts import (
    ArtifactReference,
    AttemptId,
    ContractRecord,
    Identifier,
    ShardId,
    validate_absolute_path,
)


class ClientOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ClientResult(ContractRecord):
    """Semantic Data Designer outcome independent of engine-internal result types."""

    run_id: Identifier
    shard_id: ShardId
    attempt_id: AttemptId
    completed_at: datetime
    requested_records: PositiveInt
    actual_records: NonNegativeInt | None
    outcome: ClientOutcome
    dataset_path: str | None = None
    early_shutdown: bool | None = None
    requested_resume_mode: Literal["never", "always", "if_possible"]
    effective_resume_mode: Literal["never", "always"] | None = None
    candidate_output_manifest: ArtifactReference | None = None
    error_code: Identifier | None = None
    redacted_message: Annotated[str, StringConstraints(max_length=512)] | None = None

    @field_validator("completed_at")
    @classmethod
    def validate_completed_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("completed_at must be timezone-aware UTC")
        return value

    @field_validator("dataset_path")
    @classmethod
    def validate_dataset_path(cls, value: str | None) -> str | None:
        return None if value is None else validate_absolute_path(value)

    @field_validator("redacted_message")
    @classmethod
    def validate_message(cls, value: str | None) -> str | None:
        if value is not None and any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("redacted_message must not contain control characters")
        return value

    @model_validator(mode="after")
    def validate_outcome(self) -> ClientResult:
        if self.actual_records is not None and self.actual_records > self.requested_records:
            raise ValueError("actual_records must not exceed requested_records")
        if self.requested_resume_mode != "if_possible" and self.effective_resume_mode not in {
            None,
            self.requested_resume_mode,
        }:
            raise ValueError("effective resume mode must match a fixed requested mode")
        if self.outcome is not ClientOutcome.FAILED:
            if self.early_shutdown is None or self.effective_resume_mode is None:
                raise ValueError("non-failed client results require resume and early-shutdown facts")
        if self.outcome is ClientOutcome.COMPLETE:
            if self.actual_records != self.requested_records:
                raise ValueError("complete client results require the requested record count")
            self._require_success_artifacts()
        elif self.outcome is ClientOutcome.PARTIAL:
            if self.actual_records is None or not 0 < self.actual_records < self.requested_records:
                raise ValueError("partial client results require a positive partial record count")
            self._require_success_artifacts()
        else:
            if self.candidate_output_manifest is not None:
                raise ValueError("failed client results cannot reference a candidate output manifest")
            if self.error_code is None:
                raise ValueError("failed client results require error_code")
        return self

    def _require_success_artifacts(self) -> None:
        if self.dataset_path is None or self.candidate_output_manifest is None:
            raise ValueError("successful client results require dataset and candidate manifest paths")
        if self.error_code is not None or self.redacted_message is not None:
            raise ValueError("successful client results cannot contain failure details")
        shard_root = f"/runs/{self.run_id}/shards/{self.shard_id}"
        if self.effective_resume_mode == "never":
            expected_dataset = f"{shard_root}/attempts/{self.attempt_id}/dataset"
        else:
            expected_dataset = f"{shard_root}/dataset"
        if not self.dataset_path.endswith(expected_dataset):
            raise ValueError("dataset path must match the run, shard, attempt, and resume policy")
        expected_manifest = f"{shard_root}/attempts/{self.attempt_id}/output-manifest.json"
        if not self.candidate_output_manifest.path.endswith(expected_manifest):
            raise ValueError("candidate output reference must match the run, shard, and attempt")
