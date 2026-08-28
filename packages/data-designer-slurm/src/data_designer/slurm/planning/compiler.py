# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal pure deterministic Slurm plan compilation."""

from __future__ import annotations

import posixpath

from pydantic import ValidationError

from data_designer.slurm._errors import format_validation_error
from data_designer.slurm.contracts import (
    ArtifactReference,
    RecordRange,
    ResumeWorkspace,
    compute_serialized_json_sha256,
)
from data_designer.slurm.planning.errors import SlurmPlanCompilationError, SlurmPlanContractError
from data_designer.slurm.planning.models import (
    PlannedShard,
    PortClaim,
    ResolvedClient,
    ResolvedDeployment,
    ResolvedSlurmRunPlan,
    ResolvedTopology,
)
from data_designer.slurm.planning.resolution import (
    EffectiveDataDesignerSlurmConfig,
    validate_effective_slurm_config,
)
from data_designer.slurm.planning.validation import validate_resolved_plan

__all__: list[str] = []

_LOGICAL_ENDPOINT_PORT = 17000
_HTTP_PORT = 18000
_RENDEZVOUS_PORT = 19000
_PORT_RANGE_SIZE = 1000


class SlurmRunCompiler:
    """Compile one fully resolved configuration without ambient state or I/O."""

    @staticmethod
    def compile(effective: EffectiveDataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return one immutable deterministic execution plan."""
        try:
            validate_effective_slurm_config(effective)
            deployments = _compile_deployments(effective)
            client = _compile_client(effective, deployments)
            _validate_port_claims(effective, client, deployments)
            plan = ResolvedSlurmRunPlan(
                schema_version=1,
                run_id=effective.run_id,
                package_version=effective.package_version,
                authored_config=ArtifactReference(
                    path=posixpath.join(_run_root(effective), "authored-config.json"),
                    sha256=effective.authored.compute_sha256(),
                ),
                selected_profile=effective.selected_profile,
                resolved_gpus_per_node=effective.resolved_gpus_per_node,
                builder=effective.builder,
                invocation=effective.invocation,
                client=client,
                deployments=deployments,
                array_tasks=effective.authored.array_tasks,
                shards=_compile_shards(effective),
                submission=effective.submission,
                output=effective.output,
                container_mounts=tuple(effective.selected_profile.profile.container_mounts),
                runtime_bundle=effective.runtime_bundle,
            )
            return validate_resolved_plan(
                effective.authored,
                effective.dependency_lock,
                plan,
                builder_payload=effective.builder_payload,
            )
        except (SlurmPlanCompilationError, SlurmPlanContractError):
            raise
        except ValidationError as error:
            message = format_validation_error(error, subject="Slurm plan compilation")
            raise SlurmPlanCompilationError(message) from None
        except ValueError as error:
            raise SlurmPlanCompilationError(str(error)) from None


def _compile_deployments(
    effective: EffectiveDataDesignerSlurmConfig,
) -> tuple[ResolvedDeployment, ...]:
    if len(effective.authored.deployments) > _PORT_RANGE_SIZE:
        raise SlurmPlanCompilationError("deployment count exceeds the compiler-owned logical endpoint port range")
    resolved: list[ResolvedDeployment] = []
    next_node_index = 0
    for index, (authored, image) in enumerate(
        zip(effective.authored.deployments, effective.deployment_images, strict=True)
    ):
        tensor_parallel = authored.topology.tensor_parallel
        if effective.resolved_gpus_per_node % tensor_parallel:
            raise SlurmPlanCompilationError("tensor_parallel must divide resolved GPUs per node")
        replicas_per_group = effective.resolved_gpus_per_node // tensor_parallel
        if replicas_per_group > _PORT_RANGE_SIZE:
            raise SlurmPlanCompilationError("replica lanes exceed the compiler-owned deployment port range")
        node_group_count = authored.resources.nodes // authored.topology.nodes_per_replica
        replica_count = node_group_count * replicas_per_group
        topology = ResolvedTopology(
            tensor_parallel=tensor_parallel,
            nodes_per_replica=authored.topology.nodes_per_replica,
            pipeline_parallel=authored.topology.nodes_per_replica,
            node_group_count=node_group_count,
            replicas_per_node_group=replicas_per_group,
            replica_count=replica_count,
            gpus_per_replica=tensor_parallel * authored.topology.nodes_per_replica,
        )
        deployment_id = f"deployment-{index:05d}"
        node_indices = tuple(range(next_node_index, next_node_index + authored.resources.nodes))
        ports = _compile_deployment_ports(deployment_id, node_indices, topology)
        resolved.append(
            ResolvedDeployment(
                deployment_id=deployment_id,
                authored=authored,
                served_model_name=authored.served_model_name or authored.model,
                image=image,
                node_indices=node_indices,
                gpus_per_node=effective.resolved_gpus_per_node,
                topology=topology,
                ports=ports,
            )
        )
        next_node_index += authored.resources.nodes
    return tuple(resolved)


def _compile_deployment_ports(
    deployment_id: str,
    node_indices: tuple[int, ...],
    topology: ResolvedTopology,
) -> tuple[PortClaim, ...]:
    http: list[PortClaim] = []
    rendezvous: list[PortClaim] = []
    for group_index in range(topology.node_group_count):
        head = node_indices[group_index * topology.nodes_per_replica]
        for lane_index in range(topology.replicas_per_node_group):
            replica_index = group_index * topology.replicas_per_node_group + lane_index
            http.append(
                PortClaim(
                    name=f"{deployment_id}-http-{replica_index:05d}",
                    role="http",
                    node_index=head,
                    port=_HTTP_PORT + lane_index,
                )
            )
            if topology.nodes_per_replica > 1:
                rendezvous.append(
                    PortClaim(
                        name=f"{deployment_id}-rendezvous-{replica_index:05d}",
                        role="rendezvous",
                        node_index=head,
                        port=_RENDEZVOUS_PORT + lane_index,
                    )
                )
    return tuple(http + rendezvous)


def _compile_client(
    effective: EffectiveDataDesignerSlurmConfig,
    deployments: tuple[ResolvedDeployment, ...],
) -> ResolvedClient:
    host_node_index = deployments[0].node_indices[0]
    ports = tuple(
        PortClaim(
            name=f"{deployment.deployment_id}-logical-endpoint",
            role="logical_endpoint",
            node_index=host_node_index,
            port=_LOGICAL_ENDPOINT_PORT + index,
        )
        for index, deployment in enumerate(deployments)
    )
    return ResolvedClient(
        authored=effective.authored.client,
        image=effective.client_image,
        dependency_lock=ArtifactReference(
            path=posixpath.join(_run_root(effective), "dependency-lock.json"),
            sha256=effective.dependency_lock.compute_sha256(),
        ),
        host_node_index=host_node_index,
        gpu_count=0,
        ports=ports,
    )


def _validate_port_claims(
    effective: EffectiveDataDesignerSlurmConfig,
    client: ResolvedClient,
    deployments: tuple[ResolvedDeployment, ...],
) -> None:
    ports = client.ports + tuple(port for deployment in deployments for port in deployment.ports)
    addresses = tuple((port.node_index, port.port) for port in ports)
    if len(addresses) != len(set(addresses)):
        raise SlurmPlanCompilationError("compiler-owned port claims collide on one node")
    otel_port = effective.invocation.effective_run_config.get("otel_metrics_port")
    if type(otel_port) is int and (client.host_node_index, otel_port) in addresses:
        raise SlurmPlanCompilationError("client OTEL metrics port collides with a compiler-owned port")


def _compile_shards(effective: EffectiveDataDesignerSlurmConfig) -> tuple[PlannedShard, ...]:
    count = effective.authored.array_tasks.count
    requested = effective.authored.invocation.num_records
    floor_count = requested // count
    start = 0
    shards: list[PlannedShard] = []
    for index in range(count):
        record_count = requested - floor_count * (count - 1) if index == count - 1 else floor_count
        end = start + record_count
        shard_id = f"shard-{index:05d}"
        shard_root = posixpath.join(_run_root(effective), "shards", shard_id)
        record_range = RecordRange(start_index=start, end_index_exclusive=end)
        partition = None
        seed_path = effective.invocation.effective_input_bindings.seed_path
        if seed_path is not None:
            partition = ArtifactReference(
                path=posixpath.join(shard_root, "input-partition.json"),
                sha256=compute_serialized_json_sha256(
                    {
                        "record_range": record_range.model_dump(mode="json"),
                        "seed_path": seed_path,
                    }
                ),
            )
        shards.append(
            PlannedShard(
                shard_id=shard_id,
                shard_index=index,
                array_task_index=index,
                record_range=record_range,
                input_partition=partition,
                resume_workspace=ResumeWorkspace(path=posixpath.join(shard_root, "dataset")),
            )
        )
        start = end
    return tuple(shards)


def _run_root(effective: EffectiveDataDesignerSlurmConfig) -> str:
    return posixpath.join(effective.selected_profile.profile.workspace_root, "runs", effective.run_id)
