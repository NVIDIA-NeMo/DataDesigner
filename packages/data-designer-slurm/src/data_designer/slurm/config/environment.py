# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed environment bindings and secret-shape validation."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Annotated, Literal

from pydantic import Field, JsonValue, StringConstraints, field_validator

from data_designer.slurm.contracts import AuthoredConfig, validate_plain_text
from data_designer.slurm.types import EnvironmentName

__all__ = [
    "EnvironmentBinding",
    "LiteralEnvironmentBinding",
    "SecretRef",
]


class LiteralEnvironmentBinding(AuthoredConfig):
    """Literal, non-secret environment value materialized for a Slurm process.

    Attributes:
        type: Discriminator for serialized environment bindings.
        value: Plain-text value. Secret-shaped environment names cannot use this binding.
    """

    type: Literal["literal"]
    value: Annotated[str, StringConstraints(max_length=4096)]

    @field_validator("value")
    @classmethod
    def validate_value(cls, value: str) -> str:
        return validate_plain_text(value, field_name="environment value")


class SecretRef(AuthoredConfig):
    """Reference to a login-environment variable containing secret material.

    The referenced value is materialized only at the runtime boundary and is never
    persisted into authored or resolved contracts.

    Attributes:
        type: Discriminator for serialized environment bindings.
        environment: Name of the source variable in the trusted login environment.
    """

    type: Literal["secret"]
    environment: EnvironmentName


EnvironmentBinding = Annotated[
    LiteralEnvironmentBinding | SecretRef,
    Field(discriminator="type"),
]


def is_secret_bearing_name(value: str) -> bool:
    """Return whether a Slurm-owned name conventionally carries secret material."""
    segments = _secret_name_segments(value)
    return bool(
        _SECRET_NAME_PARTS.intersection(segments)
        or {"access", "key"}.issubset(segments)
        or segments[-1] in {"auth", "key"}
    )


def validate_environment_bindings(values: Mapping[EnvironmentName, EnvironmentBinding]) -> None:
    """Reject literal values for environment variables that look secret-bearing."""
    if any(
        is_secret_bearing_name(name) and isinstance(binding, LiteralEnvironmentBinding)
        for name, binding in values.items()
    ):
        raise ValueError("secret-shaped environment names require external secret references")


def validate_no_plaintext_secrets(value: JsonValue, *, field_name: str) -> None:
    """Reject non-null values stored under recursively secret-bearing JSON keys."""
    if isinstance(value, dict):
        for key, child in value.items():
            if is_secret_bearing_json_key(key) and child is not None:
                raise ValueError(f"{field_name} must not contain plaintext values under secret-bearing keys")
            validate_no_plaintext_secrets(child, field_name=field_name)
    elif isinstance(value, list):
        for child in value:
            validate_no_plaintext_secrets(child, field_name=field_name)


def is_secret_bearing_json_key(value: str) -> bool:
    """Return whether a persisted JSON key conventionally carries secret material."""
    segments = _secret_name_segments(value)
    return tuple(segments[-2:]) not in _NON_SECRET_KEY_SUFFIXES and is_secret_bearing_name(value)


_SECRET_NAME_PARTS = frozenset({"credential", "credentials", "password", "secret", "token"})
_NON_SECRET_KEY_SUFFIXES = frozenset(
    {
        ("foreign", "key"),
        ("idempotency", "key"),
        ("partition", "key"),
        ("primary", "key"),
        ("sort", "key"),
    }
)


def _secret_name_segments(value: str) -> list[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
    return normalized.split("_")
