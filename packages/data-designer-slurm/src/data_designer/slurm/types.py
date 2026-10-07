# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constrained scalar types for Slurm configuration and records."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BeforeValidator, Field, StringConstraints

Identifier = Annotated[
    str,
    StringConstraints(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]*$",
    ),
]


def _normalize_partition_selection(value: object) -> object:
    if isinstance(value, list):
        if not value or not all(isinstance(name, str) for name in value):
            raise ValueError("partitions must be a nonempty list of names")
        value = ",".join(value)
    if isinstance(value, str) and len(set(value.split(","))) != len(value.split(",")):
        raise ValueError("partition names must be unique")
    return value


PartitionSelection = Annotated[
    str,
    BeforeValidator(_normalize_partition_selection),
    StringConstraints(
        min_length=1,
        max_length=1024,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}(?:,[A-Za-z0-9][A-Za-z0-9._-]{0,127})*$",
    ),
]
ShardId = Annotated[str, StringConstraints(pattern=r"^shard-[0-9]{5,}$")]
AttemptId = Annotated[str, StringConstraints(pattern=r"^attempt-[0-9]{4,}$")]
SchemaVersion = Literal[1]
EnvironmentName = Annotated[str, StringConstraints(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")]
Sha256Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Duration = Annotated[str, StringConstraints(pattern=r"^[1-9][0-9]*(?:s|m|h|d)$")]
NonNegativeDuration = Annotated[str, StringConstraints(pattern=r"^(?:0|[1-9][0-9]*)(?:s|m|h|d)$")]
NetworkPort = Annotated[int, Field(ge=1024, le=65535)]

__all__ = [
    "AttemptId",
    "Duration",
    "EnvironmentName",
    "Identifier",
    "NetworkPort",
    "NonNegativeDuration",
    "PartitionSelection",
    "SchemaVersion",
    "Sha256Digest",
    "ShardId",
]
