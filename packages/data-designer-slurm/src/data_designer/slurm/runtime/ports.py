# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local port isolation for shared Slurm nodes."""

from __future__ import annotations

from collections.abc import Mapping

from data_designer.slurm.planning import PortClaim, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.network import get_service_ports, verify_service_ports
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.resolver import resolve_vllm_server

_PORTS_PER_GPU = 256


def resolve_allocation_deployments(
    context: AllocationContext,
    environment: Mapping[str, str],
) -> tuple[ResolvedVllmServerDeployment, ...]:
    """Resolve deployments with ports isolated to one scheduler array element."""
    plan = resolve_allocation_plan(context.plan, environment)
    return tuple(resolve_vllm_server(plan, item.deployment_id) for item in plan.deployments)


def allocation_ports(context: AllocationContext, environment: Mapping[str, str]) -> tuple[int, ...]:
    """Return every allocation-local port in deterministic claim order."""
    plan = resolve_allocation_plan(context.plan, environment)
    return tuple(port.port for port in plan.client.ports) + tuple(
        port.port for deployment in plan.deployments for port in deployment.ports
    )


def resolve_allocation_plan(
    plan: ResolvedSlurmRunPlan,
    environment: Mapping[str, str],
) -> ResolvedSlurmRunPlan:
    """Return a plan with ports bound to the allocation's assigned GPUs."""
    if not plan.deployments:
        return plan
    claims = plan.client.ports + tuple(port for deployment in plan.deployments for port in deployment.ports)
    if plan.selected_profile.profile.gpu_request_mode == "visible":
        verify_service_ports(tuple(claim.port for claim in claims))
        return plan
    gpu_ids = _parse_gpu_ids(environment.get("SLURM_JOB_GPUS"))
    offset = min(gpu_ids) * _PORTS_PER_GPU
    block = get_service_ports()[offset : offset + _PORTS_PER_GPU]
    if len(block) != _PORTS_PER_GPU:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "allocated GPU index exceeds the available non-ephemeral allocation port range",
        )
    claims_by_node: dict[int, set[int]] = {}
    for claim in claims:
        claims_by_node.setdefault(claim.node_index, set()).add(claim.port)
    reserved_by_node: dict[int, set[int]] = {}
    otel_port = plan.invocation.effective_run_config.get("otel_metrics_port")
    if type(otel_port) is int and otel_port in block:
        reserved_by_node[plan.client.host_node_index] = {otel_port}
    mapping: dict[tuple[int, int], int] = {}
    for node_index, planned_ports in claims_by_node.items():
        reserved = reserved_by_node.get(node_index, set())
        available = iter(port for port in block if port not in reserved)
        for planned_port in sorted(planned_ports):
            next_port = next(available, None)
            if next_port is None:
                raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation port range is exhausted")
            mapping[(node_index, planned_port)] = next_port

    def remap(port: PortClaim) -> PortClaim:
        return port.model_copy(update={"port": mapping[(port.node_index, port.port)]})

    client = plan.client.model_copy(update={"ports": tuple(remap(port) for port in plan.client.ports)})
    deployments = tuple(
        deployment.model_copy(update={"ports": tuple(remap(port) for port in deployment.ports)})
        for deployment in plan.deployments
    )
    return plan.model_copy(update={"client": client, "deployments": deployments})


def _parse_gpu_ids(value: str | None) -> tuple[int, ...]:
    if value is None:
        gpu_ids = ()
    else:
        fields = value.split(",")
        gpu_ids = tuple(int(field) for field in fields if field.isascii() and field.isdigit())
        if len(gpu_ids) != len(fields):
            gpu_ids = ()
    if not gpu_ids or len(gpu_ids) != len(set(gpu_ids)):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "scheduler environment 'SLURM_JOB_GPUS' is unavailable or invalid",
        )
    return gpu_ids


__all__ = ["allocation_ports", "resolve_allocation_deployments", "resolve_allocation_plan"]
