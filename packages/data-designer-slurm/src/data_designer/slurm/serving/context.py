# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Planner-to-serving resolution context."""

from __future__ import annotations

from pydantic import NonNegativeInt, model_validator

from data_designer.slurm.contracts import ContractValue
from data_designer.slurm.planning.models import PortClaim, ResolvedClient, ResolvedDeployment


class ServerResolutionContext(ContractValue):
    """Planner-supplied placement and endpoint claims for one deployment."""

    placement: ResolvedDeployment
    client_host_node_index: NonNegativeInt
    logical_endpoint: PortClaim

    @model_validator(mode="after")
    def validate_logical_endpoint(self) -> ServerResolutionContext:
        expected_name = f"{self.placement.deployment_id}-logical-endpoint"
        if self.logical_endpoint.role != "logical_endpoint" or self.logical_endpoint.name != expected_name:
            raise ValueError("server resolution requires the deployment's logical endpoint claim")
        if self.logical_endpoint.node_index != self.client_host_node_index:
            raise ValueError("logical endpoint claim must use the resolved client host")
        if any(
            port.node_index == self.logical_endpoint.node_index and port.port == self.logical_endpoint.port
            for port in self.placement.ports
        ):
            raise ValueError("logical endpoint claim must not collide with a deployment port")
        return self

    @classmethod
    def from_plan(cls, placement: ResolvedDeployment, client: ResolvedClient) -> ServerResolutionContext:
        """Join one deployment placement to its client-owned logical endpoint claim."""
        expected_name = f"{placement.deployment_id}-logical-endpoint"
        matches = tuple(port for port in client.ports if port.name == expected_name)
        if len(matches) != 1:
            raise ValueError("resolved client must contain exactly one logical endpoint for the deployment")
        return cls(
            placement=placement,
            client_host_node_index=client.host_node_index,
            logical_endpoint=matches[0],
        )
