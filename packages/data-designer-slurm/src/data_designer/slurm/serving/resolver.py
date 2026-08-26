# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure deployment-owned server resolution for Slurm execution."""

from __future__ import annotations

from data_designer.slurm.config.images import ServingImageInspection
from data_designer.slurm.config.run import ServerDeploymentConfig
from data_designer.slurm.config.vllm import VllmServerConfig
from data_designer.slurm.planning.models import PortClaim, ResolvedDeployment
from data_designer.slurm.serving.compatibility import UnsupportedServingRuntimeError, resolve_vllm_compatibility
from data_designer.slurm.serving.context import ServerResolutionContext
from data_designer.slurm.serving.deployment import ResolvedServerDeployment
from data_designer.slurm.serving.endpoints import (
    ResolvedBackendEndpoint,
    ResolvedLogicalEndpoint,
    ResolvedReadinessProbe,
)
from data_designer.slurm.serving.processes import (
    VllmLaunchPolicy,
    VllmProcessRole,
    VllmProcessSpec,
    VllmRendezvousSpec,
)

_DURATION_FACTORS = {"s": 1, "m": 60, "h": 3600, "d": 86400}


class ServerResolutionError(ValueError):
    """Raised when planner inputs cannot produce a supported server specification."""


def resolve_server(
    deployment: ServerDeploymentConfig,
    context: ServerResolutionContext,
) -> ResolvedServerDeployment:
    """Resolve one typed server declaration against planner-owned placement."""
    if deployment != context.placement.authored:
        raise ServerResolutionError("server declaration does not match its planner-owned placement")
    match deployment.server:
        case VllmServerConfig():
            return _resolve_vllm(deployment, context.placement, context.logical_endpoint)
    raise ServerResolutionError(f"unsupported server type: {deployment.server.type!r}")


def _resolve_vllm(
    deployment: ServerDeploymentConfig,
    placement: ResolvedDeployment,
    logical_port: PortClaim,
) -> ResolvedServerDeployment:
    server = deployment.server
    inspection = placement.image.inspection.inspection
    if not isinstance(inspection, ServingImageInspection) or inspection.server_type != server.type:
        raise ServerResolutionError("resolved serving image does not match the declared vLLM server")
    try:
        compatibility = resolve_vllm_compatibility(inspection.runtime_version)
    except UnsupportedServingRuntimeError as error:
        raise ServerResolutionError(str(error)) from error
    if placement.topology.pipeline_parallel > 1 and not compatibility.supports_multi_node:
        raise ServerResolutionError("inspected vLLM version does not support the resolved multi-node topology")
    if server.enable_expert_parallel and placement.topology.pipeline_parallel > 1:
        raise ServerResolutionError("multi-node expert parallel is not supported in v1")

    http_ports = tuple(port for port in placement.ports if port.role == "http")
    rendezvous_ports = tuple(port for port in placement.ports if port.role == "rendezvous")
    processes: list[VllmProcessSpec] = []
    backends: list[ResolvedBackendEndpoint] = []
    probes: list[ResolvedReadinessProbe] = []
    nodes_per_replica = placement.topology.nodes_per_replica
    replicas_per_group = placement.topology.replicas_per_node_group
    tensor_parallel = placement.topology.tensor_parallel

    for node_group_index in range(placement.topology.node_group_count):
        group_start = node_group_index * nodes_per_replica
        group_nodes = placement.node_indices[group_start : group_start + nodes_per_replica]
        for lane_index in range(replicas_per_group):
            replica_index = node_group_index * replicas_per_group + lane_index
            http_port = http_ports[replica_index]
            backend_id = f"{placement.deployment_id}-backend-{replica_index:05d}"
            backends.append(
                ResolvedBackendEndpoint(
                    backend_id=backend_id,
                    replica_index=replica_index,
                    node_group_index=node_group_index,
                    lane_index=lane_index,
                    node_index=http_port.node_index,
                    port=http_port.port,
                    served_model_name=placement.served_model_name,
                )
            )
            probes.append(
                ResolvedReadinessProbe(
                    probe_id=f"{placement.deployment_id}-readiness-{replica_index:05d}",
                    backend_id=backend_id,
                    node_index=http_port.node_index,
                    port=http_port.port,
                    path=server.readiness_path,
                    deadline_seconds=_duration_seconds(server.startup_timeout),
                )
            )
            rendezvous = _resolve_rendezvous(
                placement,
                rendezvous_ports,
                node_group_index=node_group_index,
                lane_index=lane_index,
                replica_index=replica_index,
            )
            gpu_start = lane_index * tensor_parallel
            gpu_indices = tuple(range(gpu_start, gpu_start + tensor_parallel))
            launch_delay = _launch_delay_seconds(server, replica_index)
            for pipeline_rank, node_index in enumerate(group_nodes):
                is_head = pipeline_rank == 0
                processes.append(
                    VllmProcessSpec(
                        process_id=(f"{placement.deployment_id}-replica-{replica_index:05d}-rank-{pipeline_rank:05d}"),
                        deployment_id=placement.deployment_id,
                        replica_index=replica_index,
                        node_group_index=node_group_index,
                        lane_index=lane_index,
                        pipeline_rank=pipeline_rank,
                        node_index=node_index,
                        gpu_indices=gpu_indices,
                        role=VllmProcessRole.API_SERVER if is_head else VllmProcessRole.FOLLOWER,
                        tensor_parallel=tensor_parallel,
                        pipeline_parallel=placement.topology.pipeline_parallel,
                        http_port=http_port.port if is_head else None,
                        rendezvous=rendezvous,
                        launch_delay_seconds=launch_delay,
                    )
                )

    return ResolvedServerDeployment(
        deployment_id=placement.deployment_id,
        server_type="vllm",
        model_alias=deployment.model_alias,
        model=deployment.model,
        served_model_name=placement.served_model_name,
        image=placement.image,
        executable_path=inspection.executable_path,
        node_indices=placement.node_indices,
        gpus_per_node=placement.gpus_per_node,
        topology=placement.topology,
        compatibility=compatibility,
        launch_policy=VllmLaunchPolicy(
            startup_timeout_seconds=_duration_seconds(server.startup_timeout),
            distributed_init_timeout_seconds=_duration_seconds(server.distributed_init_timeout),
            lead_boot_standoff_seconds=_duration_seconds(server.lead_boot_standoff),
            rank_launch_stagger_seconds=_duration_seconds(server.rank_launch_stagger),
            readiness_path=server.readiness_path,
            enable_expert_parallel=server.enable_expert_parallel,
            queue_backpressure=server.queue_backpressure,
            extra_args=tuple(server.extra_args),
            environment=dict(server.environment),
        ),
        processes=tuple(processes),
        readiness_probes=tuple(probes),
        backend_endpoints=tuple(backends),
        logical_endpoint=ResolvedLogicalEndpoint(
            endpoint_id=logical_port.name,
            model_alias=deployment.model_alias,
            served_model_name=placement.served_model_name,
            node_index=logical_port.node_index,
            port=logical_port.port,
            backend_ids=tuple(backend.backend_id for backend in backends),
        ),
    )


def _resolve_rendezvous(
    placement: ResolvedDeployment,
    rendezvous_ports: tuple[PortClaim, ...],
    *,
    node_group_index: int,
    lane_index: int,
    replica_index: int,
) -> VllmRendezvousSpec | None:
    if placement.topology.pipeline_parallel == 1:
        return None
    port = rendezvous_ports[replica_index]
    return VllmRendezvousSpec(
        node_group_index=node_group_index,
        lane_index=lane_index,
        master_node_index=port.node_index,
        port=port.port,
        timeout_seconds=_duration_seconds(placement.authored.server.distributed_init_timeout),
    )


def _launch_delay_seconds(server: VllmServerConfig, replica_index: int) -> int:
    if replica_index == 0:
        return 0
    return _duration_seconds(server.lead_boot_standoff) + replica_index * _duration_seconds(server.rank_launch_stagger)


def _duration_seconds(value: str) -> int:
    return int(value[:-1]) * _DURATION_FACTORS[value[-1]]
