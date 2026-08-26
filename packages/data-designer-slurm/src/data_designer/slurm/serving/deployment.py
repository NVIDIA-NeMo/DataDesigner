# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Self-validating resolved server deployment record."""

from __future__ import annotations

from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.images import ServingImageInspection
from data_designer.slurm.contracts import (
    ContractValue,
    validate_absolute_path,
    validate_plain_text,
)
from data_designer.slurm.planning.models import ResolvedImage, ResolvedTopology
from data_designer.slurm.serving.endpoints import (
    ResolvedBackendEndpoint,
    ResolvedLogicalEndpoint,
    ResolvedReadinessProbe,
)
from data_designer.slurm.serving.vllm import VllmLaunchPolicy, VllmProcessSpec
from data_designer.slurm.types import Identifier


class ResolvedServerDeployment(ContractValue):
    """Complete runtime-neutral serving specification for one deployment."""

    deployment_id: Identifier
    server_type: Literal["vllm"]
    model_alias: str
    model: str
    served_model_name: str
    image: ResolvedImage
    executable_path: str
    node_indices: tuple[NonNegativeInt, ...] = Field(min_length=1)
    gpus_per_node: PositiveInt
    topology: ResolvedTopology
    launch_policy: VllmLaunchPolicy
    processes: tuple[VllmProcessSpec, ...] = Field(min_length=1)
    readiness_probes: tuple[ResolvedReadinessProbe, ...] = Field(min_length=1)
    backend_endpoints: tuple[ResolvedBackendEndpoint, ...] = Field(min_length=1)
    logical_endpoint: ResolvedLogicalEndpoint
    failure_policy: Literal["coordinated"] = "coordinated"

    _executable_path_is_absolute = field_validator("executable_path")(validate_absolute_path)

    @field_validator("model", "served_model_name")
    @classmethod
    def validate_model_names(cls, value: str) -> str:
        return validate_plain_text(value, field_name="model name")

    @model_validator(mode="after")
    def validate_resolution(self) -> ResolvedServerDeployment:
        inspection = self.image.inspection.inspection
        if not isinstance(inspection, ServingImageInspection) or inspection.server_type != self.server_type:
            raise ValueError("resolved server image inspection must match the server type")
        if inspection.executable_path != self.executable_path:
            raise ValueError("resolved executable path must match the serving image inspection")
        if self.node_indices != tuple(sorted(set(self.node_indices))):
            raise ValueError("resolved server nodes must be sorted and unique")
        if len(self.node_indices) % self.topology.nodes_per_replica:
            raise ValueError("resolved server nodes must divide evenly into replica groups")
        if self.gpus_per_node % self.topology.tensor_parallel:
            raise ValueError("resolved server GPUs must divide evenly into tensor-parallel lanes")
        expected_topology = ResolvedTopology(
            tensor_parallel=self.topology.tensor_parallel,
            nodes_per_replica=self.topology.nodes_per_replica,
            pipeline_parallel=self.topology.nodes_per_replica,
            node_group_count=len(self.node_indices) // self.topology.nodes_per_replica,
            replicas_per_node_group=self.gpus_per_node // self.topology.tensor_parallel,
            replica_count=(len(self.node_indices) // self.topology.nodes_per_replica)
            * (self.gpus_per_node // self.topology.tensor_parallel),
            gpus_per_replica=self.topology.tensor_parallel * self.topology.nodes_per_replica,
        )
        if self.topology != expected_topology:
            raise ValueError("resolved server topology must match its node and GPU resources")
        if self.launch_policy.enable_expert_parallel and self.topology.pipeline_parallel > 1:
            raise ValueError("multi-node expert parallel is not supported in v1")
        if self.logical_endpoint.model_alias != self.model_alias:
            raise ValueError("logical endpoint model alias must match the deployment")
        if self.logical_endpoint.served_model_name != self.served_model_name:
            raise ValueError("logical endpoint served model name must match the deployment")
        if self.logical_endpoint.endpoint_id != f"{self.deployment_id}-logical-endpoint":
            raise ValueError("logical endpoint ID must match the deployment")

        replica_indices = tuple(endpoint.replica_index for endpoint in self.backend_endpoints)
        expected_replicas = tuple(range(self.topology.replica_count))
        if replica_indices != expected_replicas:
            raise ValueError("backend endpoints must use complete ordered replica identities")
        backend_ids = tuple(endpoint.backend_id for endpoint in self.backend_endpoints)
        expected_backend_ids = tuple(
            f"{self.deployment_id}-backend-{replica_index:05d}" for replica_index in expected_replicas
        )
        if backend_ids != expected_backend_ids:
            raise ValueError("backend endpoint IDs must match the deployment and replica order")
        if self.logical_endpoint.backend_ids != backend_ids:
            raise ValueError("logical endpoint backends must match the resolved backend order")
        if len({process.process_id for process in self.processes}) != len(self.processes):
            raise ValueError("resolved process IDs must be unique")

        expected_process_count = self.topology.replica_count * self.topology.pipeline_parallel
        if len(self.processes) != expected_process_count:
            raise ValueError("resolved process count must match replica and pipeline topology")
        process_identities = tuple((process.replica_index, process.pipeline_rank) for process in self.processes)
        expected_process_identities = tuple(
            (replica_index, pipeline_rank)
            for replica_index in range(self.topology.replica_count)
            for pipeline_rank in range(self.topology.pipeline_parallel)
        )
        if process_identities != expected_process_identities:
            raise ValueError("resolved processes must use complete ordered replica and pipeline identities")

        probes_by_backend = {probe.backend_id: probe for probe in self.readiness_probes}
        if len(self.readiness_probes) != len(self.backend_endpoints) or tuple(probes_by_backend) != backend_ids:
            raise ValueError("readiness probes must match the resolved backend order")
        for endpoint in self.backend_endpoints:
            self._validate_replica(endpoint, probes_by_backend[endpoint.backend_id])
        self._validate_network_addresses()
        return self

    def _validate_replica(self, endpoint: ResolvedBackendEndpoint, probe: ResolvedReadinessProbe) -> None:
        expected_group = endpoint.replica_index // self.topology.replicas_per_node_group
        expected_lane = endpoint.replica_index % self.topology.replicas_per_node_group
        expected_head_node = self.node_indices[expected_group * self.topology.nodes_per_replica]
        if (
            endpoint.node_group_index != expected_group
            or endpoint.lane_index != expected_lane
            or endpoint.node_index != expected_head_node
            or endpoint.served_model_name != self.served_model_name
        ):
            raise ValueError("backend endpoint must match its resolved replica placement")
        if (
            (probe.node_index, probe.port) != (endpoint.node_index, endpoint.port)
            or probe.probe_id != f"{self.deployment_id}-readiness-{endpoint.replica_index:05d}"
            or probe.path != self.launch_policy.readiness_path
            or probe.deadline_seconds != self.launch_policy.startup_timeout_seconds
        ):
            raise ValueError("readiness probes must target their resolved backend endpoint")
        replica_processes = tuple(
            process for process in self.processes if process.replica_index == endpoint.replica_index
        )
        expected_gpu_start = expected_lane * self.topology.tensor_parallel
        expected_gpus = tuple(range(expected_gpu_start, expected_gpu_start + self.topology.tensor_parallel))
        expected_delay = (
            0
            if endpoint.replica_index == 0
            else self.launch_policy.lead_boot_standoff_seconds
            + endpoint.replica_index * self.launch_policy.rank_launch_stagger_seconds
        )
        for process in replica_processes:
            expected_node = self.node_indices[expected_group * self.topology.nodes_per_replica + process.pipeline_rank]
            expected_process_id = (
                f"{self.deployment_id}-replica-{endpoint.replica_index:05d}-rank-{process.pipeline_rank:05d}"
            )
            if (
                process.process_id != expected_process_id
                or process.deployment_id != self.deployment_id
                or process.node_group_index != expected_group
                or process.lane_index != expected_lane
                or process.node_index != expected_node
                or process.gpu_indices != expected_gpus
                or process.tensor_parallel != self.topology.tensor_parallel
                or process.pipeline_parallel != self.topology.pipeline_parallel
                or process.launch_delay_seconds != expected_delay
            ):
                raise ValueError("resolved process must match its deployment topology and launch policy")
        head = replica_processes[0]
        if (head.node_index, head.http_port) != (endpoint.node_index, endpoint.port):
            raise ValueError("replica head process must publish its resolved backend endpoint")
        if self.topology.pipeline_parallel > 1:
            rendezvous = head.rendezvous
            if rendezvous is None or any(process.rendezvous != rendezvous for process in replica_processes):
                raise ValueError("replica processes must share one rendezvous specification")
            if (
                rendezvous.master_node_index != expected_head_node
                or rendezvous.timeout_seconds != self.launch_policy.distributed_init_timeout_seconds
            ):
                raise ValueError("replica rendezvous must match its group head and distributed timeout")

    def _validate_network_addresses(self) -> None:
        rendezvous_addresses = tuple(
            (process.rendezvous.master_node_index, process.rendezvous.port)
            for process in self.processes
            if process.pipeline_rank == 0 and process.rendezvous is not None
        )
        addresses = (
            tuple((endpoint.node_index, endpoint.port) for endpoint in self.backend_endpoints)
            + ((self.logical_endpoint.node_index, self.logical_endpoint.port),)
            + rendezvous_addresses
        )
        if len(addresses) != len(set(addresses)):
            raise ValueError("resolved server network addresses must be unique")
