# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import posixpath
from collections.abc import Mapping
from typing import Annotated, Literal, TypeVar
from urllib.parse import urlsplit

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
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Duration = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(?:s|m|h|d)$")]

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _FrozenList(list[_Value]):
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


class _FrozenDict(dict[_Key, _Value]):
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


class AuthoredConfig(BaseModel):
    """Base for strict authored configuration values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def freeze_collections(cls, value: object) -> object:
        return _freeze_collections(value)

    def serialize_canonical_json(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def serialize_json(self) -> str:
        return pretty_json(self.model_dump(mode="json"))

    def compute_sha256(self) -> Sha256Digest:
        return hashlib.sha256(self.serialize_json().encode("utf-8")).hexdigest()


class ContractValue(BaseModel):
    """Base for strict immutable cross-process values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )

    @field_validator("*", mode="after")
    @classmethod
    def freeze_collections(cls, value: object) -> object:
        return _freeze_collections(value)


class ContractRecord(ContractValue):
    """Base for explicitly versioned records with stable serialization."""

    schema_version: SchemaVersion

    def serialize_canonical_json(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def serialize_json(self) -> str:
        return pretty_json(self.model_dump(mode="json"))

    def compute_sha256(self) -> Sha256Digest:
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
    if not value.startswith("/"):
        raise ValueError("path must be absolute")
    if value == "/":
        raise ValueError("path must not be the filesystem root")
    validate_plain_text(value, field_name="path")
    if ".." in value.split("/"):
        raise ValueError("path must not contain parent-directory components")
    if posixpath.normpath(value) != value:
        raise ValueError("path must be normalized")
    return value


def validate_local_config_path(value: str) -> str:
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
    if not value:
        raise ValueError(f"{field_name} must not be empty")
    if any(ord(character) < 32 or ord(character) == 127 for character in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def validate_url(value: str, *, field_name: str) -> str:
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
