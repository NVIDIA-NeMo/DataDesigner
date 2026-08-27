# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure authored-configuration resolution and deterministic plan compilation."""

from __future__ import annotations

import posixpath
from typing import Annotated

from pydantic import JsonValue, PositiveInt, StringConstraints, model_validator

from data_designer.config import RunConfig
from data_designer.slurm.config.images import ClientImageInspection, ImageKind
from data_designer.slurm.config.profiles import SelectedSlurmProfile
from data_designer.slurm.config.run import BuilderInput, DataDesignerSlurmConfig
from data_designer.slurm.contracts import (
    ArtifactReference,
    ContractValue,
    Identifier,
    RecordRange,
    ResumeWorkspace,
    compute_pretty_sha256,
    compute_sha256,
)
from data_designer.slurm.planning.models import (
    PlannedShard,
    PortClaim,
    ResolvedBuilderInput,
    ResolvedClient,
    ResolvedDependencyLock,
    ResolvedDeployment,
    ResolvedImage,
    ResolvedInvocation,
    ResolvedOutput,
    ResolvedSlurmRunPlan,
    ResolvedSubmission,
    ResolvedTopology,
    _extract_builder_aliases,
    _extract_builder_identity,
)
from data_designer.slurm.planning.validation import validate_resolved_plan

_LOGICAL_ENDPOINT_PORT = 17000
_HTTP_PORT = 18000
_RENDEZVOUS_PORT = 19000
_PORT_RANGE_SIZE = 1000
_SHARDABLE_COLUMN_TYPES = frozenset(
    {
        "embedding",
        "expression",
        "llm-code",
        "llm-judge",
        "llm-structured",
        "llm-text",
        "sampler",
        "seed-dataset",
        "validation",
    }
)
_COMPATIBILITY_RUN_DEFAULTS: dict[str, JsonValue] = {
    "buffer_size": 16384,
    "disable_early_shutdown": True,
    "display_tui": False,
    "max_conversation_correction_steps": 0,
    "max_conversation_restarts": 0,
    "otel_metrics_port": None,
    "shutdown_error_rate": 1.0,
}


class ConfigurationResolutionError(ValueError):
    """Raised when resolved inputs do not match one authored declaration."""


class PlanCompilationError(ValueError):
    """Raised when one effective configuration cannot produce a valid plan."""


class EffectiveDataDesignerSlurmConfig(ContractValue):
    """Fully materialized, side-effect-free input to the plan compiler."""

    run_id: Identifier
    package_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    authored: DataDesignerSlurmConfig
    selected_profile: SelectedSlurmProfile
    resolved_gpus_per_node: PositiveInt
    builder: ResolvedBuilderInput
    builder_payload: dict[str, JsonValue] | None = None
    invocation: ResolvedInvocation
    client_image: ResolvedImage
    deployment_images: tuple[ResolvedImage, ...]
    dependency_lock: ResolvedDependencyLock
    submission: ResolvedSubmission
    output: ResolvedOutput
    runtime_bundle: ArtifactReference

    @model_validator(mode="after")
    def validate_resolution(self) -> EffectiveDataDesignerSlurmConfig:
        workspace_root = self.selected_profile.profile.workspace_root
        run_root = posixpath.join(workspace_root, "runs", self.run_id)
        profile_gpus = self.selected_profile.profile.gpus_per_node
        if profile_gpus != "auto" and profile_gpus != self.resolved_gpus_per_node:
            raise ValueError("resolved GPU count does not match the selected profile")
        if self.client_image.kind is not ImageKind.CLIENT:
            raise ValueError("resolved client image must contain client inspection facts")
        if self.client_image.authored_ref != self.authored.client.image:
            raise ValueError("resolved client image does not match the authored reference")
        if len(self.deployment_images) != len(self.authored.deployments):
            raise ValueError("resolved serving images must match the authored deployment count")
        for deployment, image in zip(self.authored.deployments, self.deployment_images, strict=True):
            if image.kind is not ImageKind.SERVING:
                raise ValueError("resolved deployment image must contain serving inspection facts")
            if image.authored_ref != deployment.server.image:
                raise ValueError("resolved deployment image does not match the authored reference")
        if self.builder_payload is not None and self.authored.builder.source is None:
            raise ValueError("only sourced builder input may retain a resolved payload")
        if self.authored.builder.source is not None and self.builder_payload is None:
            raise ValueError("sourced builder input requires its resolved payload")
        if self.authored.builder.source is not None:
            assert self.builder_payload is not None
            validated_payload = BuilderInput(inline=self.builder_payload).inline
            assert validated_payload is not None
            aliases, referenced_aliases, digest = _extract_builder_identity(validated_payload)
            if self.builder.authored_source != self.authored.builder.source or self.builder.source is None:
                raise ValueError("resolved builder source does not match the authored input")
            expected_path = posixpath.join(run_root, "builder-config.json")
            if self.builder.source.path != expected_path:
                raise ValueError("resolved builder artifact path does not match the package-managed run")
            if self.builder.model_aliases != aliases:
                raise ValueError("resolved model aliases do not match the sourced builder payload")
            if self.builder.referenced_model_aliases != referenced_aliases:
                raise ValueError("resolved referenced aliases do not match the sourced builder payload")
            if self.builder.content_sha256 != digest:
                raise ValueError("resolved builder digest does not match the sourced builder payload")
        _validate_sharding_constraints(self.authored, builder_payload=self.builder_payload)
        expected_invocation = ResolvedInvocation(
            authored=self.authored.invocation,
            effective_run_config=_materialize_run_config(self.authored),
        )
        if self.invocation != expected_invocation:
            raise ValueError("resolved invocation does not match the authored invocation")
        expected_output = ResolvedOutput(
            root=self.authored.output.root or posixpath.join(run_root, "output"),
            format=self.authored.output.format,
            partitions=self.authored.output.partitions,
            require_exact_record_count=self.authored.output.require_exact_record_count,
        )
        if self.output != expected_output:
            raise ValueError("resolved output does not match the authored output")
        _validate_output_destination(self.output.root, workspace_root, run_root)
        if self.output.partitions > self.authored.invocation.num_records:
            raise ValueError("output partitions must not exceed requested records")
        runtime_root = posixpath.join(workspace_root, "runtime")
        if not _is_below(self.runtime_bundle.path, runtime_root) or not self.runtime_bundle.path.endswith(".tar.gz"):
            raise ValueError("runtime bundle must be a tar archive below the selected workspace runtime root")
        return self


def resolve_slurm_config(
    authored: DataDesignerSlurmConfig,
    *,
    selected_profile: SelectedSlurmProfile,
    client_image: ResolvedImage,
    deployment_images: tuple[ResolvedImage, ...],
    dependency_lock: ResolvedDependencyLock,
    runtime_bundle: ArtifactReference,
    run_id: str,
    package_version: str,
    resolved_gpus_per_node: int | None = None,
    builder_payload: dict[str, JsonValue] | None = None,
) -> EffectiveDataDesignerSlurmConfig:
    """Materialize all non-secret defaults from explicitly supplied resolved inputs."""
    try:
        gpus_per_node = _resolve_gpu_count(selected_profile, resolved_gpus_per_node)
        run_root = posixpath.join(selected_profile.profile.workspace_root, "runs", run_id)
        builder = _resolve_builder(authored, run_root=run_root, builder_payload=builder_payload)
        invocation = ResolvedInvocation(
            authored=authored.invocation,
            effective_run_config=_materialize_run_config(authored),
        )
        submission = ResolvedSubmission(
            account=authored.submission.account or selected_profile.profile.scheduler.account,
            partition=authored.submission.partition or selected_profile.profile.scheduler.partition,
            job_name=authored.submission.job_name,
            time_limit=authored.submission.time_limit,
            comment=authored.submission.comment,
        )
        output_root = authored.output.root or posixpath.join(run_root, "output")
        output = ResolvedOutput(
            root=output_root,
            format=authored.output.format,
            partitions=authored.output.partitions,
            require_exact_record_count=authored.output.require_exact_record_count,
        )
        _validate_dependency_resolution(authored, client_image, dependency_lock)
        return EffectiveDataDesignerSlurmConfig(
            run_id=run_id,
            package_version=package_version,
            authored=authored,
            selected_profile=selected_profile,
            resolved_gpus_per_node=gpus_per_node,
            builder=builder,
            builder_payload=builder_payload,
            invocation=invocation,
            client_image=client_image,
            deployment_images=deployment_images,
            dependency_lock=dependency_lock,
            submission=submission,
            output=output,
            runtime_bundle=runtime_bundle,
        )
    except ConfigurationResolutionError:
        raise
    except ValueError as error:
        raise ConfigurationResolutionError(str(error)) from error


class SlurmRunCompiler:
    """Compile one fully resolved configuration without ambient state or I/O."""

    @staticmethod
    def compile(effective: EffectiveDataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
        """Return one immutable deterministic execution plan."""
        try:
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
        except PlanCompilationError:
            raise
        except ValueError as error:
            raise PlanCompilationError(str(error)) from error


def compile_slurm_run_plan(effective: EffectiveDataDesignerSlurmConfig) -> ResolvedSlurmRunPlan:
    """Compile one effective configuration with the package-owned compiler."""
    return SlurmRunCompiler.compile(effective)


def _resolve_gpu_count(selected: SelectedSlurmProfile, resolved: int | None) -> int:
    configured = selected.profile.gpus_per_node
    if configured == "auto":
        if type(resolved) is not int or resolved <= 0:
            raise ConfigurationResolutionError("auto gpus_per_node requires one resolved positive integer")
        return resolved
    if resolved is not None and resolved != configured:
        raise ConfigurationResolutionError("resolved GPU count conflicts with the selected profile")
    return configured


def _resolve_builder(
    authored: DataDesignerSlurmConfig,
    *,
    run_root: str,
    builder_payload: dict[str, JsonValue] | None,
) -> ResolvedBuilderInput:
    if authored.builder.inline is not None:
        if builder_payload is not None:
            raise ConfigurationResolutionError("inline builder input must not provide a separate payload")
        aliases, referenced_aliases = _extract_builder_aliases(authored.builder.inline)
        return ResolvedBuilderInput(
            inline=authored.builder.inline,
            content_sha256=compute_sha256(authored.builder.inline),
            model_aliases=aliases,
            referenced_model_aliases=referenced_aliases,
        )
    if builder_payload is None:
        raise ConfigurationResolutionError("sourced builder input requires its resolved payload")
    validated_payload = BuilderInput(inline=builder_payload).inline
    assert validated_payload is not None
    aliases, referenced_aliases, digest = _extract_builder_identity(validated_payload)
    source = ArtifactReference(
        path=posixpath.join(run_root, "builder-config.json"),
        sha256=digest,
    )
    return ResolvedBuilderInput(
        authored_source=authored.builder.source,
        source=source,
        content_sha256=source.sha256,
        model_aliases=aliases,
        referenced_model_aliases=referenced_aliases,
    )


def _materialize_run_config(authored: DataDesignerSlurmConfig) -> dict[str, JsonValue]:
    values = dict(authored.invocation.run_config)
    authored_early_shutdown = {"disable_early_shutdown", "shutdown_error_rate", "shutdown_error_window"}.intersection(
        values
    )
    for name, value in _COMPATIBILITY_RUN_DEFAULTS.items():
        if authored_early_shutdown and name in {"disable_early_shutdown", "shutdown_error_rate"}:
            continue
        values.setdefault(name, value)
    return RunConfig.model_validate(values).model_dump(mode="json")


def _validate_sharding_constraints(
    authored: DataDesignerSlurmConfig,
    *,
    builder_payload: dict[str, JsonValue] | None,
) -> None:
    if authored.array_tasks.count == 1:
        return
    if authored.output.format != "parquet":
        raise ConfigurationResolutionError("multi-shard runs require parquet output")

    payload = authored.builder.inline if authored.builder.inline is not None else builder_payload
    assert payload is not None
    data_designer = payload.get("data_designer", payload)
    if not isinstance(data_designer, dict):
        raise ConfigurationResolutionError("builder data_designer value must be an object")
    if data_designer.get("processors"):
        raise ConfigurationResolutionError("multi-shard runs do not support global processors")
    if data_designer.get("profilers"):
        raise ConfigurationResolutionError("multi-shard runs do not support global profilers")

    seed_config = data_designer.get("seed_config")
    if isinstance(seed_config, dict):
        if seed_config.get("sampling_strategy") == "shuffle":
            raise ConfigurationResolutionError("multi-shard runs do not support shuffled seed input")
        if seed_config.get("selection_strategy") is not None:
            raise ConfigurationResolutionError("multi-shard runs do not support authored seed selection strategies")
        if authored.invocation.input_bindings.seed_path is None:
            raise ConfigurationResolutionError("multi-shard seed input requires a typed seed_path binding")

    columns = data_designer.get("columns", [])
    if not isinstance(columns, list):
        raise ConfigurationResolutionError("builder columns must be a list")
    for column in columns:
        if not isinstance(column, dict) or not isinstance(column.get("column_type"), str):
            raise ConfigurationResolutionError("multi-shard runs require known column semantics")
        column_type = column["column_type"]
        if column_type == "image":
            raise ConfigurationResolutionError("multi-shard runs do not support media output columns")
        if column_type not in _SHARDABLE_COLUMN_TYPES:
            raise ConfigurationResolutionError(
                "multi-shard runs do not support custom, plugin, or unknown column semantics"
            )
        if column_type == "validation" and column.get("validator_type") == "local_callable":
            raise ConfigurationResolutionError("multi-shard runs do not support local callable validators")


def _validate_dependency_resolution(
    authored: DataDesignerSlurmConfig,
    client_image: ResolvedImage,
    dependency_lock: ResolvedDependencyLock,
) -> None:
    inspection = client_image.inspection.inspection
    if not isinstance(inspection, ClientImageInspection):
        raise ConfigurationResolutionError("resolved client image lacks dependency inspection facts")
    if dependency_lock.client_image_sha256 != client_image.sha256:
        raise ConfigurationResolutionError("dependency lock does not match the resolved client image")
    if dependency_lock.python_abi != inspection.python_abi:
        raise ConfigurationResolutionError("dependency lock Python ABI does not match the client image")
    if dependency_lock.image_distributions != inspection.distributions:
        raise ConfigurationResolutionError("dependency lock inventory does not match the client image")
    requirements = authored.client.dependencies.requirements
    if requirements is not None:
        if dependency_lock.authored_source is not None or dependency_lock.source is not None:
            raise ConfigurationResolutionError("inline requirements cannot resolve from an authored lock file")
        if dependency_lock.authored_requirements != tuple(requirements):
            raise ConfigurationResolutionError("dependency lock requirements do not match authored requirements")
    elif dependency_lock.authored_source != authored.client.dependencies.lock_file or dependency_lock.source is None:
        raise ConfigurationResolutionError("dependency lock source does not match the authored lock file")


def _compile_deployments(
    effective: EffectiveDataDesignerSlurmConfig,
) -> tuple[ResolvedDeployment, ...]:
    if len(effective.authored.deployments) > _PORT_RANGE_SIZE:
        raise PlanCompilationError("deployment count exceeds the compiler-owned logical endpoint port range")
    resolved: list[ResolvedDeployment] = []
    next_node_index = 0
    for index, (authored, image) in enumerate(
        zip(effective.authored.deployments, effective.deployment_images, strict=True)
    ):
        tensor_parallel = authored.topology.tensor_parallel
        if effective.resolved_gpus_per_node % tensor_parallel:
            raise PlanCompilationError("tensor_parallel must divide resolved GPUs per node")
        replicas_per_group = effective.resolved_gpus_per_node // tensor_parallel
        if replicas_per_group > _PORT_RANGE_SIZE:
            raise PlanCompilationError("replica lanes exceed the compiler-owned deployment port range")
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
        raise PlanCompilationError("compiler-owned port claims collide on one node")
    otel_port = effective.invocation.effective_run_config.get("otel_metrics_port")
    if type(otel_port) is int and (client.host_node_index, otel_port) in addresses:
        raise PlanCompilationError("client OTEL metrics port collides with a compiler-owned port")


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
        seed_path = effective.authored.invocation.input_bindings.seed_path
        if seed_path is not None:
            partition = ArtifactReference(
                path=posixpath.join(shard_root, "input-partition.json"),
                sha256=compute_pretty_sha256(
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


def _validate_output_destination(output_root: str, workspace_root: str, run_root: str) -> None:
    if not _is_below(output_root, workspace_root):
        raise ConfigurationResolutionError("output root must be below the selected workspace_root")
    reserved = tuple(posixpath.join(workspace_root, name) for name in ("images", "runtime", "benchmarks"))
    if any(_paths_overlap(output_root, path) for path in reserved):
        raise ConfigurationResolutionError("output root must not overlap package-managed workspace state")
    runs_root = posixpath.join(workspace_root, "runs")
    run_output_root = posixpath.join(run_root, "output")
    if _paths_overlap(output_root, runs_root) and not (
        output_root == run_output_root or _is_below(output_root, run_output_root)
    ):
        raise ConfigurationResolutionError("output root must not overlap another package-managed run")


def _run_root(effective: EffectiveDataDesignerSlurmConfig) -> str:
    return posixpath.join(effective.selected_profile.profile.workspace_root, "runs", effective.run_id)


def _is_below(path: str, root: str) -> bool:
    return path != root and posixpath.commonpath((path, root)) == root


def _paths_overlap(left: str, right: str) -> bool:
    return left == right or _is_below(left, right) or _is_below(right, left)
