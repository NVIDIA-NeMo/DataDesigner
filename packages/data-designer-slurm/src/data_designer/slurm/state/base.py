# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import posixpath
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, StringConstraints, field_validator

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StateValue(BaseModel):
    """Base for strict, immutable values nested within state records."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )


class StateRecord(StateValue):
    """Base for immutable, strictly versioned Slurm state records."""

    schema_version: Literal[1]

    def serialize_canonical_json(self) -> bytes:
        """Serialize the record to a stable compact JSON representation."""
        return json.dumps(
            self.model_dump(mode="json"),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def serialize_json(self) -> str:
        """Serialize the record to deterministic, human-readable JSON."""
        return (
            json.dumps(
                self.model_dump(mode="json"),
                allow_nan=False,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def compute_sha256(self) -> Sha256Digest:
        """Compute the digest of the exact bytes written by ``serialize_json``."""
        return hashlib.sha256(self.serialize_json().encode("utf-8")).hexdigest()


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


def validate_absolute_path(value: str) -> str:
    """Validate a normalized, absolute POSIX path below the filesystem root."""
    if not value.startswith("/"):
        raise ValueError("path must be absolute")
    if value.startswith("//"):
        raise ValueError("path must have exactly one leading slash")
    if value == "/":
        raise ValueError("path must not be the filesystem root")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("path must not contain control characters")
    if ".." in value.split("/"):
        raise ValueError("path must not contain parent-directory components")
    if posixpath.normpath(value) != value:
        raise ValueError("path must be normalized")
    return value


def validate_relative_path(value: str) -> str:
    """Validate a normalized relative POSIX path without parent traversal."""
    if not value or value.startswith("/"):
        raise ValueError("path must be a non-empty relative path")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError("path must not contain control characters")
    if ".." in value.split("/"):
        raise ValueError("path must not contain parent-directory components")
    if posixpath.normpath(value) != value or value == ".":
        raise ValueError("path must be normalized")
    return value


class ArtifactReference(StateValue):
    """Immutable reference to an on-disk artifact and its content digest."""

    path: str
    sha256: Sha256Digest

    _path_is_safe = field_validator("path")(validate_absolute_path)


class SchedulerIdentity(StateValue):
    """Slurm array job and task identity assigned to one attempt."""

    array_job_id: PositiveInt
    array_task_id: NonNegativeInt
