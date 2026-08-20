# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timedelta

from pydantic import NonNegativeInt, PositiveInt

from data_designer.slurm.contracts import (
    ArtifactReference,
    ContractRecord,
    ContractValue,
    Identifier,
    Sha256Digest,
    validate_absolute_path,
    validate_relative_path,
)

StateValue = ContractValue
StateRecord = ContractRecord


def validate_utc_timestamp(value: datetime) -> datetime:
    """Validate that a timestamp is timezone-aware and expressed in UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include timezone information")
    if value.utcoffset() != timedelta(0):
        raise ValueError("timestamp must be in UTC")
    return value


def validate_optional_utc_timestamp(value: datetime | None) -> datetime | None:
    """Validate an optional timestamp when it is present."""
    if value is None:
        return None
    return validate_utc_timestamp(value)


class SchedulerIdentity(StateValue):
    """Slurm array job and task identity assigned to one attempt."""

    array_job_id: PositiveInt
    array_task_id: NonNegativeInt


__all__ = [
    "ArtifactReference",
    "Identifier",
    "SchedulerIdentity",
    "Sha256Digest",
    "StateRecord",
    "StateValue",
    "validate_absolute_path",
    "validate_optional_utc_timestamp",
    "validate_relative_path",
    "validate_utc_timestamp",
]
