# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import posixpath
from typing import Annotated, Literal

from pydantic import Field, JsonValue, NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.config import RunConfig
from data_designer.slurm._contracts import (
    ContractRecord,
    ContractValue,
    Identifier,
    Sha256Digest,
    compute_sha256,
    validate_absolute_path,
)
from data_designer.slurm.config.images import (
    DistributionName,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
)
from data_designer.slurm.config.profiles import ContainerMount, SelectedSlurmProfile
from data_designer.slurm.config.run import (
    ArrayTasksConfig,
    ClientConfig,
    InvocationConfig,
    ServerDeploymentConfig,
)


class ArtifactReference(ContractValue):
    """Immutable reference to a persisted artifact and its digest."""

    path: str
    sha256: Sha256Digest

    _path_is_absolute = field_validator("path")(validate_absolute_path)


class ResolvedImage(ContractValue):
    """Immutable SQSH path and the digest-bound inspection that approved it."""

    authored_ref: ImageRef
    path: str
    sha256: Sha256Digest
    inspection: ImageInspectionRecord

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        validate_absolute_path(value)
        if not value.endswith(".sqsh"):
            raise ValueError("resolved image path must end in .sqsh")
        return value

    @model_validator(mode="after")
    def validate_image(self) -> ResolvedImage:
        if self.inspection.sqsh_sha256 != self.sha256:
            raise ValueError("image inspection digest does not match the resolved SQSH")
        if self.authored_ref.path is not None and self.authored_ref.path != self.path:
            raise ValueError("resolved image path does not match the authored path")
        return self

    @property
    def kind(self) -> ImageKind:
        return self.inspection.inspection.kind


class LockedPackage(ContractValue):
    name: DistributionName
    version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    artifact: ArtifactReference

    @model_validator(mode="after")
    def validate_artifact(self) -> LockedPackage:
        if not self.artifact.path.endswith(".whl"):
            raise ValueError("locked overlay artifacts must be wheels")
        return self


class ResolvedDependencyLock(ContractRecord):
    """Immutable client dependency resolution against one fixed image inventory."""

    resolver_version: Identifier
    python_abi: Identifier
    client_image_sha256: Sha256Digest
    authored_requirements: tuple[str, ...]
    image_distributions: tuple[InstalledDistribution, ...]
    overlay_packages: tuple[LockedPackage, ...]

    @model_validator(mode="after")
    def validate_packages(self) -> ResolvedDependencyLock:
        image_names = tuple(distribution.name for distribution in self.image_distributions)
        overlay_names = tuple(package.name for package in self.overlay_packages)
        if image_names != tuple(sorted(image_names)) or overlay_names != tuple(sorted(overlay_names)):
            raise ValueError("dependency lock distributions must be sorted by normalized name")
        if len(image_names) != len(set(image_names)) or len(overlay_names) != len(set(overlay_names)):
            raise ValueError("dependency lock distribution names must be unique")
        overlap = set(image_names).intersection(overlay_names)
        if overlap:
            raise ValueError(f"overlay packages overlap image-owned distributions: {', '.join(sorted(overlap))}")
        return self


class ResolvedBuilderInput(ContractValue):
    authored_source: str | None = None
    source: ArtifactReference | None = None
    inline: dict[str, JsonValue] | None = None
    content_sha256: Sha256Digest

    @model_validator(mode="after")
    def validate_input(self) -> ResolvedBuilderInput:
        if (self.source is None) == (self.inline is None):
            raise ValueError("resolved builder requires exactly one of source or inline")
        if self.source is None:
            if self.authored_source is not None:
                raise ValueError("inline builder input cannot contain authored_source")
            expected_digest = compute_sha256(self.inline)
        else:
            if self.authored_source is None:
                raise ValueError("resolved builder source requires authored_source")
            expected_digest = self.source.sha256
        if self.content_sha256 != expected_digest:
            raise ValueError("builder content digest does not match the resolved input")
        return self


class ResolvedInvocation(ContractValue):
    authored: InvocationConfig
    effective_run_config: dict[str, JsonValue]

    @field_validator("effective_run_config", mode="before")
    @classmethod
    def materialize_run_config(cls, value: object) -> dict[str, JsonValue]:
        return RunConfig.model_validate(value).model_dump(mode="json")


class PortClaim(ContractValue):
    name: Identifier
    node_index: NonNegativeInt
    port: Annotated[int, Field(ge=1024, le=65535)]


class ResolvedTopology(ContractValue):
    tensor_parallel: PositiveInt
    nodes_per_replica: PositiveInt
    pipeline_parallel: PositiveInt
    node_group_count: PositiveInt
    replicas_per_node_group: PositiveInt
    replica_count: PositiveInt
    gpus_per_replica: PositiveInt


class ResolvedDeployment(ContractValue):
    deployment_id: Identifier
    authored: ServerDeploymentConfig
    image: ResolvedImage
    node_indices: tuple[NonNegativeInt, ...] = Field(min_length=1)
    gpus_per_node: PositiveInt
    topology: ResolvedTopology
    ports: tuple[PortClaim, ...] = ()

    @model_validator(mode="after")
    def validate_deployment(self) -> ResolvedDeployment:
        if self.image.kind is not ImageKind.SERVING:
            raise ValueError("server deployments require serving images")
        if self.image.authored_ref != self.authored.server.image:
            raise ValueError("resolved serving image does not match the authored image reference")
        if len(self.node_indices) != self.authored.resources.nodes:
            raise ValueError("deployment placement must contain exactly the requested node count")
        if self.node_indices != tuple(sorted(set(self.node_indices))):
            raise ValueError("deployment node indices must be sorted and unique")
        if self.gpus_per_node % self.authored.topology.tensor_parallel:
            raise ValueError("tensor_parallel must divide resolved GPUs per node")
        expected = ResolvedTopology(
            tensor_parallel=self.authored.topology.tensor_parallel,
            nodes_per_replica=self.authored.topology.nodes_per_replica,
            pipeline_parallel=self.authored.topology.nodes_per_replica,
            node_group_count=self.authored.resources.nodes // self.authored.topology.nodes_per_replica,
            replicas_per_node_group=self.gpus_per_node // self.authored.topology.tensor_parallel,
            replica_count=(self.authored.resources.nodes // self.authored.topology.nodes_per_replica)
            * (self.gpus_per_node // self.authored.topology.tensor_parallel),
            gpus_per_replica=self.authored.topology.tensor_parallel * self.authored.topology.nodes_per_replica,
        )
        if self.topology != expected:
            raise ValueError("resolved topology does not match deployment resources")
        if any(port.node_index not in self.node_indices for port in self.ports):
            raise ValueError("deployment port claims must use deployment nodes")
        return self


class ResolvedClient(ContractValue):
    authored: ClientConfig
    image: ResolvedImage
    dependency_lock: ArtifactReference
    host_node_index: NonNegativeInt
    gpu_count: Literal[0]

    @model_validator(mode="after")
    def validate_client(self) -> ResolvedClient:
        if self.image.kind is not ImageKind.CLIENT:
            raise ValueError("Data Designer client requires a client image")
        if self.image.authored_ref != self.authored.image:
            raise ValueError("resolved client image does not match the authored image reference")
        return self


class PlannedShard(ContractValue):
    shard_id: Identifier
    shard_index: NonNegativeInt
    array_task_index: NonNegativeInt
    start_index: NonNegativeInt
    end_index_exclusive: PositiveInt
    requested_records: PositiveInt
    resume_workspace: str

    _workspace_is_absolute = field_validator("resume_workspace")(validate_absolute_path)

    @model_validator(mode="after")
    def validate_range(self) -> PlannedShard:
        if self.end_index_exclusive <= self.start_index:
            raise ValueError("shard end_index_exclusive must be greater than start_index")
        if self.requested_records != self.end_index_exclusive - self.start_index:
            raise ValueError("shard requested_records must match its record range")
        return self


class ResolvedSubmission(ContractValue):
    account: Identifier | None = None
    partition: Identifier | None = None
    job_name: Identifier
    time_limit: str
    comment: str | None = None


class ResolvedOutput(ContractValue):
    root: str
    format: Literal["parquet", "jsonl", "csv"]
    partitions: PositiveInt
    require_exact_record_count: bool

    _root_is_absolute = field_validator("root")(validate_absolute_path)


class ResolvedSlurmRunPlan(ContractRecord):
    """Immutable allocation input consumed without ambient configuration."""

    plan_id: Identifier
    package_version: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    authored_config: ArtifactReference
    selected_profile: SelectedSlurmProfile
    resolved_gpus_per_node: PositiveInt
    builder: ResolvedBuilderInput
    invocation: ResolvedInvocation
    client: ResolvedClient
    deployments: tuple[ResolvedDeployment, ...] = Field(min_length=1)
    array_tasks: ArrayTasksConfig
    shards: tuple[PlannedShard, ...] = Field(min_length=1)
    submission: ResolvedSubmission
    output: ResolvedOutput
    container_mounts: tuple[ContainerMount, ...] = ()
    runtime_bundle: ArtifactReference

    @model_validator(mode="after")
    def validate_plan(self) -> ResolvedSlurmRunPlan:
        profile = self.selected_profile.profile
        if profile.gpus_per_node != "auto" and profile.gpus_per_node != self.resolved_gpus_per_node:
            raise ValueError("resolved GPU count does not match the selected profile")
        if any(deployment.gpus_per_node != self.resolved_gpus_per_node for deployment in self.deployments):
            raise ValueError("every deployment must use the resolved profile GPU count")
        if tuple(profile.container_mounts) != self.container_mounts:
            raise ValueError("plan mount mappings must match the selected profile")

        deployment_ids = tuple(deployment.deployment_id for deployment in self.deployments)
        aliases = tuple(deployment.authored.model_alias for deployment in self.deployments)
        if len(deployment_ids) != len(set(deployment_ids)):
            raise ValueError("resolved deployment IDs must be unique")
        if len(aliases) != len(set(aliases)):
            raise ValueError("resolved deployment aliases must be unique")

        node_indices = tuple(index for deployment in self.deployments for index in deployment.node_indices)
        if node_indices != tuple(range(len(node_indices))):
            raise ValueError("deployment nodes must be disjoint and contiguous in authored order")
        if self.client.host_node_index != self.deployments[0].node_indices[0]:
            raise ValueError("client must be colocated on the first node of the first deployment")

        port_keys = tuple((port.node_index, port.port) for deployment in self.deployments for port in deployment.ports)
        if len(port_keys) != len(set(port_keys)):
            raise ValueError("plan port claims must be unique per node")

        self._validate_shards()
        if not _is_below(self.output.root, profile.workspace_root):
            raise ValueError("resolved output root must be below the selected workspace_root")
        return self

    def _validate_shards(self) -> None:
        if len(self.shards) != self.array_tasks.count:
            raise ValueError("plan must contain exactly one shard per array task")
        requested_records = self.invocation.authored.num_records
        floor_count = requested_records // self.array_tasks.count
        expected_start = 0
        for index, shard in enumerate(self.shards):
            if shard.shard_index != index or shard.array_task_index != index:
                raise ValueError("shards must use complete ordered zero-based identities")
            if shard.start_index != expected_start:
                raise ValueError("shard record ranges must be contiguous")
            expected_count = (
                requested_records - floor_count * (self.array_tasks.count - 1)
                if index == self.array_tasks.count - 1
                else floor_count
            )
            if shard.requested_records != expected_count:
                raise ValueError("shards must use deterministic floor/remainder record counts")
            expected_start = shard.end_index_exclusive
        if expected_start != requested_records:
            raise ValueError("shard record ranges must cover the requested records")


def _is_below(path: str, root: str) -> bool:
    return path != root and posixpath.commonpath((path, root)) == root
