# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.slurm._contracts import ContractRecord, Identifier, validate_absolute_path
from data_designer.slurm.planning import ArtifactReference


class ClientOutcome(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    FAILED = "failed"


class ClientResult(ContractRecord):
    """Semantic Data Designer outcome independent of engine-internal result types."""

    run_id: Identifier
    shard_id: Identifier
    attempt_id: Identifier
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
