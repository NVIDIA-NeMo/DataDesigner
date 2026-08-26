# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared constrained scalar types for Slurm configuration and records."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

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
NonNegativeDuration = Annotated[str, StringConstraints(pattern=r"^(?:0|[1-9][0-9]*)(?:s|m|h|d)$")]
NetworkPort = Annotated[int, Field(ge=1024, le=65535)]

__all__ = [
    "AttemptId",
    "Duration",
    "EnvironmentName",
    "Identifier",
    "ModelAlias",
    "NetworkPort",
    "NonNegativeDuration",
    "SchemaVersion",
    "Sha256Digest",
    "ShardId",
]
