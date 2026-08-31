# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared immutable contracts for Slurm planning and runtime state."""

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Mapping
from typing import TypeVar
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    field_validator,
    model_validator,
)

from data_designer.slurm.types import (
    AttemptId,
    Duration,
    EnvironmentName,
    Identifier,
    NetworkPort,
    NonNegativeDuration,
    SchemaVersion,
    Sha256Digest,
    ShardId,
)

# TODO: Remove these compatibility exports after Stage 2 branches import shared scalars from slurm.types.
ModelAlias = str

_KeyT = TypeVar("_KeyT")
_ValueT = TypeVar("_ValueT")


class _FrozenList(list[_ValueT]):
    """List that retains JSON compatibility without exposing mutation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("frozen list cannot be modified")

    __delitem__ = _immutable
    __iadd__ = _immutable
    __imul__ = _immutable
    __setitem__ = _immutable
    append = _immutable
    clear = _immutable
    extend = _immutable
    insert = _immutable
    pop = _immutable
    remove = _immutable
    reverse = _immutable
    sort = _immutable


class _FrozenDict(dict[_KeyT, _ValueT]):
    """Dictionary that retains JSON compatibility without exposing mutation."""

    def _immutable(self, *args: object, **kwargs: object) -> None:
        del args, kwargs
        raise TypeError("frozen dictionary cannot be modified")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


def _freeze_collections(value: object) -> object:
    if isinstance(value, Mapping):
        return _FrozenDict({key: _freeze_collections(item) for key, item in value.items()})
    if isinstance(value, list):
        return _FrozenList(_freeze_collections(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_collections(item) for item in value)
    return value


class ContractValue(BaseModel):
    """Base for strict immutable values shared across Slurm boundaries."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        hide_input_in_errors=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def freeze_collections(cls, value: object) -> object:
        return _freeze_collections(value)


class AuthoredConfig(ContractValue):
    """Base for strict authored configuration values."""

    def serialize_canonical_json(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def serialize_json(self) -> str:
        return pretty_json(self.model_dump(mode="json"))

    def compute_sha256(self) -> Sha256Digest:
        """Compute the digest of the exact bytes written by ``serialize_json``."""
        return hashlib.sha256(self.serialize_json().encode("utf-8")).hexdigest()


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


def compute_canonical_json_sha256(value: object) -> Sha256Digest:
    """Compute the canonical JSON digest of a JSON-compatible value."""
    return hashlib.sha256(canonical_json(value)).hexdigest()


def compute_serialized_json_sha256(value: object) -> Sha256Digest:
    """Compute the digest of deterministic persisted JSON bytes."""
    return hashlib.sha256(pretty_json(value).encode("utf-8")).hexdigest()


def derive_managed_assets_path(workspace_root: str) -> str:
    """Derive the default managed-assets path from a workspace root."""
    return posixpath.join(workspace_root, "managed-assets")


def is_path_below(path: str, root: str) -> bool:
    """Return whether a path is strictly below a root."""
    return path != root and posixpath.commonpath((path, root)) == root


def paths_overlap(left: str, right: str) -> bool:
    """Return whether either path contains the other."""
    return left == right or is_path_below(left, right) or is_path_below(right, left)


# TODO: Remove after in-flight Stage 2 branches adopt the explicit canonical-JSON digest name.
compute_sha256 = compute_canonical_json_sha256


def validate_absolute_path(value: str) -> str:
    """Validate a normalized, absolute POSIX path below the filesystem root."""
    if not value.startswith("/"):
        raise ValueError("path must be absolute")
    if value.startswith("//"):
        raise ValueError("path must have exactly one leading slash")
    if value == "/":
        raise ValueError("path must not be the filesystem root")
    validate_plain_text(value, field_name="path")
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


def validate_local_config_path(value: str) -> str:
    """Validate a local JSON or YAML configuration path."""
    validate_plain_text(value, field_name="path")
    if "://" in value:
        raise ValueError("builder and config sources must be local paths")
    if ".." in value.split("/"):
        raise ValueError("path must not contain parent-directory components")
    normalized = posixpath.normpath(value)
    if posixpath.splitext(normalized)[1] not in {".json", ".yaml", ".yml"}:
        raise ValueError("config path must end in .json, .yaml, or .yml")
    return normalized


def validate_plain_text(value: str, *, field_name: str) -> str:
    """Reject empty text and control characters at persisted boundaries."""
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def validate_url(value: str, *, field_name: str) -> str:
    """Validate an HTTP(S) URL with a valid host and port."""
    validate_plain_text(value, field_name=field_name)
    try:
        parsed = urlsplit(value)
        parsed.port
    except ValueError as error:
        raise ValueError(f"{field_name} must be an HTTP(S) URL with a valid host and port") from error
    if (
        parsed.scheme not in {"http", "https"}
        or parsed.hostname is None
        or any(character.isspace() for character in value)
    ):
        raise ValueError(f"{field_name} must be an HTTP(S) URL")
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
    "AuthoredConfig",
    "ContractRecord",
    "ContractValue",
    "Duration",
    "EnvironmentName",
    "Identifier",
    "ModelAlias",
    "NetworkPort",
    "NonNegativeDuration",
    "RecordRange",
    "ResumeWorkspace",
    "SchemaVersion",
    "Sha256Digest",
    "ShardId",
    "canonical_json",
    "compute_canonical_json_sha256",
    "compute_serialized_json_sha256",
    "compute_sha256",
    "derive_managed_assets_path",
    "is_path_below",
    "paths_overlap",
    "pretty_json",
    "validate_absolute_path",
    "validate_local_config_path",
    "validate_plain_text",
    "validate_relative_path",
    "validate_url",
]
