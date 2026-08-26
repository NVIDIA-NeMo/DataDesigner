# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed environment bindings and secret-shape validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, field_validator

from data_designer.slurm.contracts import AuthoredConfig, EnvironmentName, validate_plain_text

__all__ = [
    "EnvironmentBinding",
    "LiteralEnvironmentBinding",
    "SecretRef",
    "contains_secret_key",
    "is_secret_name",
    "validate_environment_bindings",
]


class LiteralEnvironmentBinding(AuthoredConfig):
    type: Literal["literal"]
    value: Annotated[str, StringConstraints(max_length=4096)]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return validate_plain_text(value, field_name="environment value")


class SecretRef(AuthoredConfig):
    type: Literal["secret"]
    environment: EnvironmentName


EnvironmentBinding = Annotated[
    LiteralEnvironmentBinding | SecretRef,
    Field(discriminator="type"),
]


def contains_secret_key(value: object) -> bool:
    """Return whether a nested mapping contains a non-null secret-shaped field."""
    if isinstance(value, Mapping):
        return any(
            (_is_secret_payload_name(str(key)) and item is not None) or contains_secret_key(item)
            for key, item in value.items()
        )
    if isinstance(value, list | tuple):
        return any(contains_secret_key(item) for item in value)
    return False


def is_secret_name(value: str) -> bool:
    """Return whether a field or environment name conventionally carries a secret."""
    segments = _secret_name_segments(value)
    return bool(
        _SECRET_NAME_PARTS.intersection(segments)
        or {"access", "key"}.issubset(segments)
        or segments[-1] in {"auth", "key"}
    )


def validate_environment_bindings(
    values: dict[EnvironmentName, EnvironmentBinding],
) -> dict[EnvironmentName, EnvironmentBinding]:
    """Reject literal values for environment variables that look secret-bearing."""
    literal_secrets = [
        name
        for name, binding in values.items()
        if is_secret_name(name) and isinstance(binding, LiteralEnvironmentBinding)
    ]
    if literal_secrets:
        raise ValueError("secret-shaped environment names require external secret references")
    return values


_NON_SECRET_PAYLOAD_KEYS = frozenset({"idempotency_key", "partition_key", "primary_key", "sort_key"})
_SECRET_NAME_PARTS = frozenset({"credential", "credentials", "password", "secret", "token"})


def _is_secret_payload_name(value: str) -> bool:
    segments = _secret_name_segments(value)
    normalized = "_".join(segments)
    return normalized not in _NON_SECRET_PAYLOAD_KEYS and is_secret_name(value)


def _secret_name_segments(value: str) -> list[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
    return normalized.split("_")
