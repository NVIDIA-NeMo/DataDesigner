# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Backend, readiness, and logical endpoint records for Slurm serving."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, model_validator

from data_designer.slurm.contracts import ContractValue
from data_designer.slurm.types import Identifier, NetworkPort


class ResolvedBackendEndpoint(ContractValue):
    """Internal HTTP endpoint published by one replica's lane head."""

    backend_id: Identifier
    replica_index: NonNegativeInt
    node_group_index: NonNegativeInt
    lane_index: NonNegativeInt
    node_index: NonNegativeInt
    port: NetworkPort
    served_model_name: str


class ResolvedReadinessProbe(ContractValue):
    """Bounded readiness probe for one lane-head backend."""

    probe_id: Identifier
    backend_id: Identifier
    node_index: NonNegativeInt
    port: NetworkPort
    path: str
    deadline_seconds: PositiveInt
    expected_status_code: Literal[200] = 200


class ResolvedLogicalEndpoint(ContractValue):
    """Client-visible endpoint aggregating one deployment's healthy backends."""

    endpoint_id: Identifier
    model_alias: str
    served_model_name: str
    node_index: NonNegativeInt
    port: NetworkPort
    backend_ids: tuple[Identifier, ...] = Field(min_length=1)
    load_balancing: Literal["least_connections"] = "least_connections"
    retry_status_codes: tuple[Literal[429], ...] = (429,)
    preserve_final_overload_response: Literal[True] = True
    require_all_backends_ready: Literal[True] = True

    @model_validator(mode="after")
    def validate_backend_ids(self) -> ResolvedLogicalEndpoint:
        if len(self.backend_ids) != len(set(self.backend_ids)):
            raise ValueError("logical endpoint backend IDs must be unique")
        if self.retry_status_codes != (429,):
            raise ValueError("logical endpoint must retry HTTP 429 responses")
        return self
