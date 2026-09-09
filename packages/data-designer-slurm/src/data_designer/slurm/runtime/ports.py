# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local port isolation for shared Slurm nodes."""

from __future__ import annotations

import hashlib

from data_designer.slurm.planning import PortClaim, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import resolve_vllm_server

_PORT_RANGE_START = 10000
_PORT_RANGE_SIZE = 20000


def resolve_allocation_deployments(context: AllocationContext) -> tuple[ResolvedVllmServerDeployment, ...]:
    """Resolve deployments with ports isolated to one scheduler array element."""
    plan = _remap_plan_ports(context)
    return tuple(resolve_vllm_server(plan, item.deployment_id) for item in plan.deployments)


def allocation_ports(context: AllocationContext) -> tuple[int, ...]:
    """Return every allocation-local port in deterministic claim order."""
    plan = _remap_plan_ports(context)
    return tuple(port.port for port in plan.client.ports) + tuple(
        port.port for deployment in plan.deployments for port in deployment.ports
    )


def _remap_plan_ports(context: AllocationContext) -> ResolvedSlurmRunPlan:
    scheduler = context.attempt.scheduler
    if scheduler is None:  # pragma: no cover - AllocationContext rejects this state
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation has no scheduler identity")
    claims = context.plan.client.ports + tuple(
        port for deployment in context.plan.deployments for port in deployment.ports
    )
    claims_by_node: dict[int, set[int]] = {}
    for claim in claims:
        claims_by_node.setdefault(claim.node_index, set()).add(claim.port)
    reserved_by_node: dict[int, set[int]] = {}
    otel_port = context.plan.invocation.effective_run_config.get("otel_metrics_port")
    if type(otel_port) is int and _PORT_RANGE_START <= otel_port < _PORT_RANGE_START + _PORT_RANGE_SIZE:
        reserved_by_node[context.plan.client.host_node_index] = {otel_port}
    mapping: dict[tuple[int, int], int] = {}
    for node_index, planned_ports in claims_by_node.items():
        seed = f"{scheduler.array_job_id}:{scheduler.array_task_id}:{node_index}".encode()
        offset = int.from_bytes(hashlib.sha256(seed).digest()[:8], "big") % _PORT_RANGE_SIZE
        reserved = reserved_by_node.get(node_index, set())
        for planned_port in sorted(planned_ports):
            for _ in range(_PORT_RANGE_SIZE):
                port = _PORT_RANGE_START + offset
                offset = (offset + 1) % _PORT_RANGE_SIZE
                if port not in reserved:
                    mapping[(node_index, planned_port)] = port
                    reserved.add(port)
                    break
            else:  # pragma: no cover - plan contracts bound claims below the allocation range
                raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation port range is exhausted")

    def remap(port: PortClaim) -> PortClaim:
        return port.model_copy(update={"port": mapping[(port.node_index, port.port)]})

    client = context.plan.client.model_copy(update={"ports": tuple(remap(port) for port in context.plan.client.ports)})
    deployments = tuple(
        deployment.model_copy(update={"ports": tuple(remap(port) for port in deployment.ports)})
        for deployment in context.plan.deployments
    )
    return context.plan.model_copy(update={"client": client, "deployments": deployments})


__all__ = ["allocation_ports", "resolve_allocation_deployments"]
