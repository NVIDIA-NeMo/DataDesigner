# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import posixpath
import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, StringConstraints

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Duration = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(?:s|m|h|d)$")]


class AuthoredConfig(BaseModel):
    """Base for strict authored configuration values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )


class ContractValue(BaseModel):
    """Base for strict immutable cross-process values."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        protected_namespaces=(),
        strict=True,
        validate_default=True,
    )


class ContractRecord(ContractValue):
    """Base for explicitly versioned records with stable serialization."""

    schema_version: Literal[1]

    def serialize_canonical_json(self) -> bytes:
        return canonical_json(self.model_dump(mode="json"))

    def serialize_json(self) -> str:
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
        return hashlib.sha256(self.serialize_canonical_json()).hexdigest()


def canonical_json(value: object) -> bytes:
    """Serialize a JSON-compatible value to stable UTF-8 bytes."""
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
    if not re.fullmatch(r"https?://[^\s]+", value):
        raise ValueError(f"{field_name} must be an HTTP(S) URL")
    return value
