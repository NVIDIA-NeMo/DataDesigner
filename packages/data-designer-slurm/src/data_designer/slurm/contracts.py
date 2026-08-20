# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable contracts for Slurm planning and runtime state."""

from __future__ import annotations

import hashlib
import json
import posixpath
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
ModelAlias = str
ShardId = Annotated[str, StringConstraints(pattern=r"^shard-[0-9]{5,}$")]
AttemptId = Annotated[str, StringConstraints(pattern=r"^attempt-[0-9]{4,}$")]
SchemaVersion = Literal[1]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class ContractValue(BaseModel):
    """Base for strict immutable values shared across Slurm boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )


class ContractRecord(ContractValue):
    """Base for immutable, explicitly versioned Slurm records."""

    schema_version: SchemaVersion

    def serialize_canonical_json(self) -> bytes:
        """Serialize the record to stable compact UTF-8 bytes."""
        return canonical_json(self.model_dump(mode="json"))

    def serialize_json(self) -> str:
        """Serialize the record to deterministic persisted text."""
        return pretty_json(self.model_dump(mode="json"))

    def compute_sha256(self) -> Sha256Digest:
        """Compute the digest of the exact bytes written by ``serialize_json``."""
        return hashlib.sha256(self.serialize_json().encode("utf-8")).hexdigest()


def canonical_json(value: object) -> bytes:
    """Serialize a JSON-compatible value to stable UTF-8 bytes."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def pretty_json(value: object) -> str:
    """Serialize a JSON-compatible value to deterministic persisted text."""
    return (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def compute_sha256(value: object) -> Sha256Digest:
    """Compute the canonical JSON digest of a JSON-compatible value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


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


class ArtifactReference(ContractValue):
    """Immutable reference to persisted file bytes and their digest."""

    path: str
    sha256: Sha256Digest

    _path_is_absolute = field_validator("path")(validate_absolute_path)


class RecordRange(ContractValue):
    """Half-open global record range assigned to one shard."""

    start_index: NonNegativeInt
    end_index_exclusive: PositiveInt

    @property
    def record_count(self) -> int:
        return self.end_index_exclusive - self.start_index

    @model_validator(mode="after")
    def validate_bounds(self) -> RecordRange:
        if self.end_index_exclusive <= self.start_index:
            raise ValueError("end_index_exclusive must be greater than start_index")
        return self


class ResumeWorkspace(ContractValue):
    """Canonical shard-owned dataset workspace."""

    path: str

    _path_is_absolute = field_validator("path")(validate_absolute_path)


__all__ = [
    "ArtifactReference",
    "AttemptId",
    "ContractRecord",
    "ContractValue",
    "Identifier",
    "ModelAlias",
    "RecordRange",
    "ResumeWorkspace",
    "SchemaVersion",
    "Sha256Digest",
    "ShardId",
    "canonical_json",
    "compute_sha256",
    "pretty_json",
    "validate_absolute_path",
    "validate_relative_path",
]
