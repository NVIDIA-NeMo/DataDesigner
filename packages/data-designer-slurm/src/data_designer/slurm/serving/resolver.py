# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure deployment-owned server resolution for Slurm execution."""

from __future__ import annotations

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version
from pydantic import ValidationError

from data_designer.slurm.config.images import ServingImageInspection
from data_designer.slurm.config.utils import convert_duration_to_seconds
from data_designer.slurm.config.vllm import VllmServerConfig
from data_designer.slurm.planning.models import PortClaim, ResolvedDeployment, ResolvedSlurmRunPlan
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.endpoints import (
    ResolvedBackendEndpoint,
    ResolvedLogicalEndpoint,
    ResolvedReadinessProbe,
)
from data_designer.slurm.serving.vllm import (
    ResolvedVllmLaunchPolicy,
    ResolvedVllmProcess,
    ResolvedVllmRendezvous,
    VllmProcessRole,
)
from data_designer.slurm.types import Identifier

_SUPPORTED_VLLM_RUNTIME_RELEASES = (
    SpecifierSet("~=0.21.0"),
    SpecifierSet("~=0.22.0"),
)


class VllmServerResolutionError(ValueError):
    """Raised when planner inputs cannot produce a supported vLLM server specification."""


def resolve_vllm_server(
    plan: ResolvedSlurmRunPlan,
    deployment_id: Identifier,
) -> ResolvedVllmServerDeployment:
    """Resolve one vLLM deployment admitted by the tested runtime matrix."""
    try:
        return _resolve_vllm_server(plan, deployment_id)
    except ValidationError as error:
        raise VllmServerResolutionError("planner inputs produced an inconsistent vLLM server specification") from error


def _resolve_vllm_server(
    plan: ResolvedSlurmRunPlan,
    deployment_id: Identifier,
) -> ResolvedVllmServerDeployment:
    deployments = tuple(deployment for deployment in plan.deployments if deployment.deployment_id == deployment_id)
    if len(deployments) != 1:
        raise VllmServerResolutionError("resolved plan must contain exactly one deployment with the requested ID")
    resolved_deployment = deployments[0]
    expected_endpoint_id = f"{resolved_deployment.deployment_id}-logical-endpoint"
    logical_endpoint_ports = tuple(port for port in plan.client.ports if port.name == expected_endpoint_id)
    if len(logical_endpoint_ports) != 1:
        raise VllmServerResolutionError("resolved client must contain exactly one logical endpoint for the deployment")
    logical_endpoint_port = logical_endpoint_ports[0]
    if logical_endpoint_port.name != expected_endpoint_id or logical_endpoint_port.role != "logical_endpoint":
        raise VllmServerResolutionError("logical endpoint port must match the resolved deployment")
    if logical_endpoint_port.node_index != plan.client.host_node_index:
        raise VllmServerResolutionError("logical endpoint port must use the resolved client host")
    if (logical_endpoint_port.node_index, logical_endpoint_port.port) in {
        (port.node_index, port.port) for port in resolved_deployment.ports
    }:
        raise VllmServerResolutionError("logical endpoint port must not collide with a deployment port")
    authored_deployment = resolved_deployment.authored
    server = authored_deployment.server
    inspection = resolved_deployment.image.inspection_facts
    if not isinstance(inspection, ServingImageInspection) or inspection.server_type != server.type:
        raise VllmServerResolutionError("resolved serving image does not match the declared vLLM server")
    _validate_vllm_runtime_version(inspection.runtime_version)
    if server.enable_expert_parallel and resolved_deployment.topology.pipeline_parallel > 1:
        raise VllmServerResolutionError("multi-node expert parallel is not supported in v1")

    http_ports = tuple(port for port in resolved_deployment.ports if port.role == "http")
    rendezvous_ports = tuple(port for port in resolved_deployment.ports if port.role == "rendezvous")
    processes: list[ResolvedVllmProcess] = []
    backends: list[ResolvedBackendEndpoint] = []
    probes: list[ResolvedReadinessProbe] = []
    nodes_per_replica = resolved_deployment.topology.nodes_per_replica
    replicas_per_node_group = resolved_deployment.topology.replicas_per_node_group
    tensor_parallel = resolved_deployment.topology.tensor_parallel

    for node_group_index in range(resolved_deployment.topology.node_group_count):
        group_start = node_group_index * nodes_per_replica
        group_nodes = resolved_deployment.node_indices[group_start : group_start + nodes_per_replica]
        for replica_index_in_node_group in range(replicas_per_node_group):
            deployment_replica_index = node_group_index * replicas_per_node_group + replica_index_in_node_group
            http_port = http_ports[deployment_replica_index]
            backend_id = f"{resolved_deployment.deployment_id}-backend-{deployment_replica_index:05d}"
            backends.append(
                ResolvedBackendEndpoint(
                    backend_id=backend_id,
                    deployment_replica_index=deployment_replica_index,
                    node_group_index=node_group_index,
                    replica_index_in_node_group=replica_index_in_node_group,
                    node_index=http_port.node_index,
                    port=http_port.port,
                    served_model_name=resolved_deployment.served_model_name,
                )
            )
            probes.append(
                ResolvedReadinessProbe(
                    probe_id=f"{resolved_deployment.deployment_id}-readiness-{deployment_replica_index:05d}",
                    backend_id=backend_id,
                    node_index=http_port.node_index,
                    port=http_port.port,
                    path=server.readiness_path,
                    deadline_seconds=convert_duration_to_seconds(server.startup_timeout),
                )
            )
            rendezvous = _resolve_rendezvous(
                resolved_deployment,
                rendezvous_ports,
                node_group_index=node_group_index,
                replica_index_in_node_group=replica_index_in_node_group,
                deployment_replica_index=deployment_replica_index,
            )
            gpu_start = replica_index_in_node_group * tensor_parallel
            gpu_indices = tuple(range(gpu_start, gpu_start + tensor_parallel))
            launch_delay = _calculate_launch_delay_seconds(server, deployment_replica_index)
            for pipeline_rank, node_index in enumerate(group_nodes):
                is_head = pipeline_rank == 0
                processes.append(
                    ResolvedVllmProcess(
                        process_id=(
                            f"{resolved_deployment.deployment_id}-replica-{deployment_replica_index:05d}"
                            f"-rank-{pipeline_rank:05d}"
                        ),
                        deployment_id=resolved_deployment.deployment_id,
                        deployment_replica_index=deployment_replica_index,
                        node_group_index=node_group_index,
                        replica_index_in_node_group=replica_index_in_node_group,
                        pipeline_rank=pipeline_rank,
                        node_index=node_index,
                        gpu_indices=gpu_indices,
                        role=VllmProcessRole.API_SERVER if is_head else VllmProcessRole.FOLLOWER,
                        tensor_parallel=tensor_parallel,
                        pipeline_parallel=resolved_deployment.topology.pipeline_parallel,
                        http_port=http_port.port if is_head else None,
                        rendezvous=rendezvous,
                        launch_delay_seconds=launch_delay,
                    )
                )

    return ResolvedVllmServerDeployment(
        deployment_id=resolved_deployment.deployment_id,
        server_type="vllm",
        model_alias=authored_deployment.model_alias,
        model=authored_deployment.model,
        served_model_name=resolved_deployment.served_model_name,
        image=resolved_deployment.image,
        executable_path=inspection.executable_path,
        node_indices=resolved_deployment.node_indices,
        gpus_per_node=resolved_deployment.gpus_per_node,
        topology=resolved_deployment.topology,
        launch_policy=ResolvedVllmLaunchPolicy(
            startup_timeout_seconds=convert_duration_to_seconds(server.startup_timeout),
            distributed_init_timeout_seconds=convert_duration_to_seconds(server.distributed_init_timeout),
            lead_boot_standoff_seconds=convert_duration_to_seconds(server.lead_boot_standoff),
            rank_launch_stagger_seconds=convert_duration_to_seconds(server.rank_launch_stagger),
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
            endpoint_id=logical_endpoint_port.name,
            model_alias=authored_deployment.model_alias,
            served_model_name=resolved_deployment.served_model_name,
            node_index=logical_endpoint_port.node_index,
            port=logical_endpoint_port.port,
            backend_ids=tuple(backend.backend_id for backend in backends),
        ),
    )


def _resolve_rendezvous(
    resolved_deployment: ResolvedDeployment,
    rendezvous_ports: tuple[PortClaim, ...],
    *,
    node_group_index: int,
    replica_index_in_node_group: int,
    deployment_replica_index: int,
) -> ResolvedVllmRendezvous | None:
    if resolved_deployment.topology.pipeline_parallel == 1:
        return None
    port = rendezvous_ports[deployment_replica_index]
    return ResolvedVllmRendezvous(
        node_group_index=node_group_index,
        replica_index_in_node_group=replica_index_in_node_group,
        master_node_index=port.node_index,
        port=port.port,
        timeout_seconds=convert_duration_to_seconds(resolved_deployment.authored.server.distributed_init_timeout),
    )


def _calculate_launch_delay_seconds(server: VllmServerConfig, deployment_replica_index: int) -> int:
    if deployment_replica_index == 0:
        return 0
    return convert_duration_to_seconds(
        server.lead_boot_standoff
    ) + deployment_replica_index * convert_duration_to_seconds(server.rank_launch_stagger)


def _validate_vllm_runtime_version(runtime_version: str) -> None:
    try:
        version = Version(runtime_version)
    except InvalidVersion as error:
        raise VllmServerResolutionError(f"unsupported vLLM runtime version {runtime_version!r}") from error
    if not any(version in supported_release for supported_release in _SUPPORTED_VLLM_RUNTIME_RELEASES):
        raise VllmServerResolutionError(
            f"unsupported vLLM runtime version {runtime_version!r}; supported release lines are 0.21.x and 0.22.x"
        )
