# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import posixpath
from typing import Annotated, Literal
from urllib.parse import urlsplit

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion
from pydantic import Field, JsonValue, NonNegativeInt, PositiveInt, StringConstraints, field_validator, model_validator

from data_designer.config import RunConfig
from data_designer.slurm.config.environment import validate_no_plaintext_secrets
from data_designer.slurm.config.images import (
    ClientImageInspection,
    ImageInspectionRecord,
    ImageKind,
    ImageRef,
    InstalledDistribution,
    ServingImageInspection,
)
from data_designer.slurm.config.profiles import ContainerMount, SelectedSlurmProfile
from data_designer.slurm.config.run import (
    ArrayTasksConfig,
    ClientConfig,
    ClientDependencies,
    InputBindings,
    InvocationConfig,
    ServerDeploymentConfig,
    SubmissionConfig,
)
from data_designer.slurm.contracts import (
    ArtifactReference,
    ContractRecord,
    ContractValue,
    DistributionName,
    Identifier,
    ModelAlias,
    RecordRange,
    ResumeWorkspace,
    Sha256Digest,
    ShardId,
    compute_serialized_json_sha256,
    derive_managed_assets_path,
    is_path_below,
    paths_overlap,
    validate_absolute_path,
    validate_local_config_path,
    validate_plain_text,
)
from data_designer.slurm.planning.builder_identity import get_declared_model_aliases
from data_designer.slurm.types import NetworkPort


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

    @property
    def inspection_facts(self) -> ClientImageInspection | ServingImageInspection:
        """Return factual image inspection data without exposing record nesting to consumers."""
        return self.inspection.inspection


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
    authored_source: str | None = None
    source: ArtifactReference | None = None
    image_distributions: tuple[InstalledDistribution, ...]
    overlay_packages: tuple[LockedPackage, ...]

    @field_validator("authored_source")
    @classmethod
    def validate_authored_source(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = validate_local_config_path(value)
        if not normalized.endswith(".json"):
            raise ValueError("dependency lock source must end in .json")
        return normalized

    @model_validator(mode="after")
    def validate_packages(self) -> ResolvedDependencyLock:
        if (self.authored_source is None) != (self.source is None):
            raise ValueError("dependency lock authored and resolved sources must be provided together")
        ClientDependencies(requirements=list(self.authored_requirements))
        image_names = tuple(distribution.name for distribution in self.image_distributions)
        overlay_names = tuple(package.name for package in self.overlay_packages)
        if image_names != tuple(sorted(image_names)) or overlay_names != tuple(sorted(overlay_names)):
            raise ValueError("dependency lock distributions must be sorted by normalized name")
        if len(image_names) != len(set(image_names)) or len(overlay_names) != len(set(overlay_names)):
            raise ValueError("dependency lock distribution names must be unique")
        overlap = set(image_names).intersection(overlay_names)
        if overlap:
            raise ValueError(f"overlay packages overlap image-owned distributions: {', '.join(sorted(overlap))}")
        image_packages = {distribution.name: distribution for distribution in self.image_distributions}
        overlay_packages = {package.name: package for package in self.overlay_packages}
        for value in self.authored_requirements:
            requirement = Requirement(value)
            name = canonicalize_name(requirement.name)
            if requirement.url is not None:
                package = overlay_packages.get(name)
                digest = urlsplit(requirement.url).fragment.removeprefix("sha256=")
                if package is None or package.artifact.sha256 != digest:
                    raise ValueError(f"direct requirement {name!r} must match one locked overlay artifact")
                continue
            package = overlay_packages.get(name) or image_packages.get(name)
            if package is None:
                raise ValueError(f"authored requirement {name!r} is missing from the dependency lock")
            try:
                satisfied = requirement.specifier.contains(package.version, prereleases=True)
            except InvalidVersion as error:
                raise ValueError(f"locked package {name!r} has an invalid version") from error
            if not satisfied:
                raise ValueError(f"locked package {name!r} does not satisfy its authored requirement")
        return self


class ResolvedBuilderInput(ContractValue):
    authored_source: str | None = None
    source: ArtifactReference | None = None
    inline: dict[str, JsonValue] | None = None
    # Both forms use deterministic persisted builder JSON bytes.
    content_sha256: Sha256Digest
    model_aliases: tuple[ModelAlias, ...]

    @model_validator(mode="after")
    def validate_input(self) -> ResolvedBuilderInput:
        if (self.source is None) == (self.inline is None):
            raise ValueError("resolved builder requires exactly one of source or inline")
        if self.source is None:
            if self.authored_source is not None:
                raise ValueError("inline builder input cannot contain authored_source")
            validate_no_plaintext_secrets(self.inline, field_name="resolved inline builder input")
            expected_digest = compute_serialized_json_sha256(self.inline)
            if self.model_aliases != get_declared_model_aliases(self.inline):
                raise ValueError("resolved model aliases do not match the inline builder")
        else:
            if self.authored_source is None:
                raise ValueError("resolved builder source requires authored_source")
            expected_digest = self.source.sha256
        if len(self.model_aliases) != len(set(self.model_aliases)):
            raise ValueError("resolved builder model aliases must be unique")
        if self.content_sha256 != expected_digest:
            raise ValueError("builder content digest does not match the resolved input")
        return self


class ResolvedInvocation(ContractValue):
    authored: InvocationConfig
    effective_input_bindings: InputBindings
    effective_run_config: dict[str, JsonValue]

    @field_validator("effective_run_config")
    @classmethod
    def validate_run_config_is_materialized(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        materialized = RunConfig.model_validate(value).model_dump(mode="json")
        if value != materialized:
            raise ValueError("effective_run_config must contain the fully materialized Data Designer RunConfig")
        return value


class PortClaim(ContractValue):
    name: Identifier
    role: Literal["http", "rendezvous", "logical_endpoint"]
    # TODO: Rename node_index to allocation_node_index after downstream Stage 2 branches converge on this contract.
    node_index: NonNegativeInt
    port: NetworkPort


class ResolvedTopology(ContractValue):
    """Resource-derived serving topology shared by planning and serving."""

    tensor_parallel: PositiveInt
    nodes_per_replica: PositiveInt
    pipeline_parallel: PositiveInt
    node_group_count: PositiveInt
    replicas_per_node_group: PositiveInt
    replica_count: PositiveInt
    gpus_per_replica: PositiveInt

    @classmethod
    def derive(
        cls,
        *,
        node_count: int,
        gpus_per_node: int,
        tensor_parallel: int,
        nodes_per_replica: int,
    ) -> ResolvedTopology:
        """Derive the canonical v1 topology from placement and GPU resources."""
        if min(node_count, gpus_per_node, tensor_parallel, nodes_per_replica) <= 0:
            raise ValueError("topology derivation inputs must be positive")
        if node_count % nodes_per_replica:
            raise ValueError("nodes_per_replica must divide the deployment node count")
        if gpus_per_node % tensor_parallel:
            raise ValueError("tensor_parallel must divide resolved GPUs per node")
        node_group_count = node_count // nodes_per_replica
        replicas_per_node_group = gpus_per_node // tensor_parallel
        return cls(
            tensor_parallel=tensor_parallel,
            nodes_per_replica=nodes_per_replica,
            pipeline_parallel=nodes_per_replica,
            node_group_count=node_group_count,
            replicas_per_node_group=replicas_per_node_group,
            replica_count=node_group_count * replicas_per_node_group,
            gpus_per_replica=tensor_parallel * nodes_per_replica,
        )


class ResolvedDeployment(ContractValue):
    deployment_id: Identifier
    authored: ServerDeploymentConfig
    served_model_name: str
    image: ResolvedImage
    node_indices: tuple[NonNegativeInt, ...] = Field(min_length=1)
    gpus_per_node: PositiveInt
    topology: ResolvedTopology
    ports: tuple[PortClaim, ...] = ()

    @model_validator(mode="after")
    def validate_deployment(self) -> ResolvedDeployment:
        validate_plain_text(self.served_model_name, field_name="served model name")
        if self.served_model_name != (self.authored.served_model_name or self.authored.model):
            raise ValueError("resolved served model name does not match the authored deployment")
        if self.image.kind is not ImageKind.SERVING:
            raise ValueError("server deployments require serving images")
        if self.image.authored_ref != self.authored.server.image:
            raise ValueError("resolved serving image does not match the authored image reference")
        if len(self.node_indices) != self.authored.resources.nodes:
            raise ValueError("deployment placement must contain exactly the requested node count")
        if self.node_indices != tuple(sorted(set(self.node_indices))):
            raise ValueError("deployment node indices must be sorted and unique")
        expected = ResolvedTopology.derive(
            node_count=len(self.node_indices),
            gpus_per_node=self.gpus_per_node,
            tensor_parallel=self.authored.topology.tensor_parallel,
            nodes_per_replica=self.authored.topology.nodes_per_replica,
        )
        if self.topology != expected:
            raise ValueError("resolved topology does not match deployment resources")
        if any(port.node_index not in self.node_indices for port in self.ports):
            raise ValueError("deployment port claims must use deployment nodes")
        names = tuple(port.name for port in self.ports)
        if len(names) != len(set(names)):
            raise ValueError("deployment port claim names must be unique")
        if any(not name.startswith(f"{self.deployment_id}-") for name in names):
            raise ValueError("deployment port claim names must use the deployment ID")
        if any(port.role == "logical_endpoint" for port in self.ports):
            raise ValueError("logical endpoint ports belong to the resolved client")

        group_heads = self.node_indices[:: self.topology.nodes_per_replica]
        expected_http_nodes = tuple(head for head in group_heads for _ in range(self.topology.replicas_per_node_group))
        http_ports = tuple(port for port in self.ports if port.role == "http")
        http_nodes = tuple(port.node_index for port in http_ports)
        if http_nodes != expected_http_nodes:
            raise ValueError("deployment requires one ordered HTTP port claim per replica lane")
        expected_http_names = tuple(f"{self.deployment_id}-http-{index:05d}" for index in range(len(http_ports)))
        if tuple(port.name for port in http_ports) != expected_http_names:
            raise ValueError("deployment HTTP port names must match their ordered replica lane")

        expected_rendezvous_nodes = (
            tuple(head for head in group_heads for _ in range(self.topology.replicas_per_node_group))
            if self.topology.nodes_per_replica > 1
            else ()
        )
        rendezvous_ports = tuple(port for port in self.ports if port.role == "rendezvous")
        rendezvous_nodes = tuple(port.node_index for port in rendezvous_ports)
        if rendezvous_nodes != expected_rendezvous_nodes:
            raise ValueError("deployment requires one ordered rendezvous port claim per multi-node replica lane")
        expected_rendezvous_names = tuple(
            f"{self.deployment_id}-rendezvous-{index:05d}" for index in range(len(rendezvous_ports))
        )
        if tuple(port.name for port in rendezvous_ports) != expected_rendezvous_names:
            raise ValueError("deployment rendezvous port names must match their ordered replica lane")
        return self


class ResolvedClient(ContractValue):
    authored: ClientConfig
    image: ResolvedImage
    dependency_lock: ArtifactReference
    host_node_index: NonNegativeInt
    gpu_count: Literal[0]
    ports: tuple[PortClaim, ...] = ()

    @model_validator(mode="after")
    def validate_client(self) -> ResolvedClient:
        if self.image.kind is not ImageKind.CLIENT:
            raise ValueError("Data Designer client requires a client image")
        if self.image.authored_ref != self.authored.image:
            raise ValueError("resolved client image does not match the authored image reference")
        if any(port.role != "logical_endpoint" for port in self.ports):
            raise ValueError("resolved client ports must be logical endpoints")
        if any(port.node_index != self.host_node_index for port in self.ports):
            raise ValueError("logical endpoint ports must use the client host")
        names = tuple(port.name for port in self.ports)
        if len(names) != len(set(names)):
            raise ValueError("logical endpoint port claim names must be unique")
        return self


class PlannedShard(ContractValue):
    shard_id: ShardId
    shard_index: NonNegativeInt
    array_task_index: NonNegativeInt
    record_range: RecordRange
    input_partition: ArtifactReference | None = None
    resume_workspace: ResumeWorkspace

    @property
    def requested_records(self) -> int:
        return self.record_range.record_count


class ResolvedSubmission(ContractValue):
    account: Identifier | None = None
    partition: Identifier | None = None
    job_name: Identifier
    time_limit: str
    comment: str | None = None

    @model_validator(mode="after")
    def validate_submission(self) -> ResolvedSubmission:
        SubmissionConfig.model_validate(self.model_dump(mode="python"))
        return self


class ResolvedOutput(ContractValue):
    root: str
    format: Literal["parquet", "jsonl", "csv"]
    partitions: PositiveInt
    require_exact_record_count: bool

    _root_is_absolute = field_validator("root")(validate_absolute_path)


class ResolvedSlurmRunPlan(ContractRecord):
    """Immutable allocation input consumed without ambient configuration."""

    run_id: Identifier
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
        if profile.gpu_request_mode == "visible" and profile.scheduler.mem_per_gpu is not None:
            raise ValueError("mem_per_gpu requires GRES GPU request mode")
        if any(deployment.gpus_per_node != self.resolved_gpus_per_node for deployment in self.deployments):
            raise ValueError("every deployment must use the resolved profile GPU count")
        if tuple(profile.container_mounts) != self.container_mounts:
            raise ValueError("plan mount mappings must match the selected profile")

        deployment_ids = tuple(deployment.deployment_id for deployment in self.deployments)
        aliases = tuple(deployment.authored.model_alias for deployment in self.deployments)
        expected_deployment_ids = tuple(f"deployment-{index:05d}" for index in range(len(self.deployments)))
        if deployment_ids != expected_deployment_ids:
            raise ValueError("resolved deployment IDs must use complete ordered zero-based identities")
        if len(aliases) != len(set(aliases)):
            raise ValueError("resolved deployment aliases must be unique")
        if set(aliases) != set(self.builder.model_aliases):
            raise ValueError("resolved deployment aliases must exactly cover Data Designer model aliases")

        node_indices = tuple(index for deployment in self.deployments for index in deployment.node_indices)
        if node_indices != tuple(range(len(node_indices))):
            raise ValueError("deployment nodes must be disjoint and contiguous in authored order")
        if self.client.host_node_index != self.deployments[0].node_indices[0]:
            raise ValueError("client must be colocated on the first node of the first deployment")
        authored_bindings = self.invocation.authored.input_bindings
        expected_bindings = InputBindings(
            seed_path=authored_bindings.seed_path,
            managed_assets_path=authored_bindings.managed_assets_path
            or derive_managed_assets_path(profile.workspace_root),
        )
        if self.invocation.effective_input_bindings != expected_bindings:
            raise ValueError("effective input bindings must match the authored and selected profile input")
        managed_assets_path = self.invocation.effective_input_bindings.managed_assets_path
        assert managed_assets_path is not None
        workspace_state = tuple(
            posixpath.join(profile.workspace_root, name) for name in ("images", "runtime", "benchmarks", "runs")
        )
        if any(paths_overlap(managed_assets_path, path) for path in workspace_state):
            raise ValueError("managed_assets_path must not overlap package-managed workspace state")
        if "non_inference_max_parallel_workers" not in self.invocation.authored.run_config:
            workers = self.invocation.effective_run_config["non_inference_max_parallel_workers"]
            if workers != 4:
                raise ValueError("default non-inference worker count must match the Data Designer default")

        expected_logical_names = tuple(
            f"{deployment.deployment_id}-logical-endpoint" for deployment in self.deployments
        )
        logical_names = tuple(port.name for port in self.client.ports)
        if logical_names != expected_logical_names:
            raise ValueError("client requires one ordered logical endpoint port per deployment")

        ports = self.client.ports + tuple(port for deployment in self.deployments for port in deployment.ports)
        port_keys = tuple((port.node_index, port.port) for port in ports)
        if len(port_keys) != len(set(port_keys)):
            raise ValueError("plan port claims must be unique per node")
        port_names = tuple(port.name for port in ports)
        if len(port_names) != len(set(port_names)):
            raise ValueError("plan port claim names must be unique")
        otel_port = self.invocation.effective_run_config.get("otel_metrics_port")
        if type(otel_port) is int and any(
            port.node_index == self.client.host_node_index and port.port == otel_port for port in ports
        ):
            raise ValueError("client OTEL metrics port collides with a plan port claim")

        run_root = posixpath.join(profile.workspace_root, "runs", self.run_id)
        if self.authored_config.path != posixpath.join(run_root, "authored-config.json"):
            raise ValueError("authored config reference must use the plan run root")
        if self.client.dependency_lock.path != posixpath.join(run_root, "dependency-lock.json"):
            raise ValueError("dependency lock reference must use the plan run root")
        runtime_root = posixpath.join(profile.workspace_root, "runtime")
        runtime_name = posixpath.basename(self.runtime_bundle.path)
        if (
            not is_path_below(self.runtime_bundle.path, runtime_root)
            or runtime_name != f"{self.runtime_bundle.sha256}.tar.gz"
        ):
            raise ValueError("runtime bundle must be a content-addressed tar archive below the workspace runtime root")
        self._validate_shards(run_root)
        if not is_path_below(self.output.root, profile.workspace_root):
            raise ValueError("resolved output root must be below the selected workspace_root")
        if paths_overlap(self.output.root, managed_assets_path):
            raise ValueError("resolved output root must not overlap managed assets")
        shards_root = posixpath.join(run_root, "shards")
        if paths_overlap(self.output.root, shards_root):
            raise ValueError("resolved output root must not overlap the run shard workspace")
        return self

    def _validate_shards(self, run_root: str) -> None:
        if len(self.shards) != self.array_tasks.count:
            raise ValueError("plan must contain exactly one shard per array task")
        requested_records = self.invocation.authored.num_records
        floor_count = requested_records // self.array_tasks.count
        expected_start = 0
        shard_ids: list[ShardId] = []
        workspace_paths: list[str] = []
        partition_paths: list[str] = []
        requires_partition = self.invocation.effective_input_bindings.seed_path is not None
        for index, shard in enumerate(self.shards):
            if shard.shard_index != index or shard.array_task_index != index:
                raise ValueError("shards must use complete ordered zero-based identities")
            if shard.shard_id != f"shard-{index:05d}":
                raise ValueError("shard IDs must match their zero-based shard index")
            if shard.record_range.start_index != expected_start:
                raise ValueError("shard record ranges must be contiguous")
            expected_count = (
                requested_records - floor_count * (self.array_tasks.count - 1)
                if index == self.array_tasks.count - 1
                else floor_count
            )
            if shard.requested_records != expected_count:
                raise ValueError("shards must use deterministic floor/remainder record counts")
            expected_workspace = posixpath.join(run_root, "shards", shard.shard_id, "dataset")
            if shard.resume_workspace.path != expected_workspace:
                raise ValueError("shard resume workspace must match the run and shard identity")
            if (shard.input_partition is not None) != requires_partition:
                raise ValueError("shard input partition presence must match the authored seed input")
            if shard.input_partition is not None:
                expected_partition = posixpath.join(run_root, "shards", shard.shard_id, "input-partition.json")
                if shard.input_partition.path != expected_partition:
                    raise ValueError("shard input partition must match the run and shard identity")
                partition_paths.append(shard.input_partition.path)
            shard_ids.append(shard.shard_id)
            workspace_paths.append(shard.resume_workspace.path)
            expected_start = shard.record_range.end_index_exclusive
        if expected_start != requested_records:
            raise ValueError("shard record ranges must cover the requested records")
        if len(shard_ids) != len(set(shard_ids)):
            raise ValueError("shard IDs must be unique")
        if len(workspace_paths) != len(set(workspace_paths)):
            raise ValueError("shard resume workspaces must be unique")
        if len(partition_paths) != len(set(partition_paths)):
            raise ValueError("shard input partitions must be unique")
