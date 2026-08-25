# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deployment-wide policy and per-process records for Slurm serving."""

from __future__ import annotations

from enum import Enum

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.run import (
    EnvironmentBinding,
    QueueBackpressureConfig,
    validate_vllm_environment_bindings,
    validate_vllm_extra_args,
    validate_vllm_readiness_path,
)
from data_designer.slurm.contracts import ContractValue, EnvironmentName, Identifier
from data_designer.slurm.serving.endpoints import NetworkPort


class VllmProcessRole(str, Enum):
    """Role of one vLLM process within a resolved replica."""

    API_SERVER = "api_server"
    FOLLOWER = "follower"


class VllmLaunchPolicy(ContractValue):
    """Resolved deployment-wide lifecycle, admission, and launch inputs."""

    startup_timeout_seconds: PositiveInt
    distributed_init_timeout_seconds: PositiveInt
    lead_boot_standoff_seconds: NonNegativeInt
    rank_launch_stagger_seconds: NonNegativeInt
    readiness_path: str
    enable_expert_parallel: bool
    queue_backpressure: QueueBackpressureConfig
    extra_args: tuple[str, ...] = ()
    environment: dict[EnvironmentName, EnvironmentBinding] = Field(default_factory=dict)

    _readiness_path_is_safe = field_validator("readiness_path")(validate_vllm_readiness_path)
    _extra_args_are_safe = field_validator("extra_args")(validate_vllm_extra_args)
    _environment_is_safe = field_validator("environment")(validate_vllm_environment_bindings)

    @model_validator(mode="after")
    def validate_timeouts(self) -> VllmLaunchPolicy:
        if self.distributed_init_timeout_seconds > self.startup_timeout_seconds:
            raise ValueError("distributed initialization timeout must not exceed startup timeout")
        return self


class VllmRendezvousSpec(ContractValue):
    """Planner-owned rendezvous inputs shared by one multi-node replica."""

    node_group_index: NonNegativeInt
    lane_index: NonNegativeInt
    master_node_index: NonNegativeInt
    port: NetworkPort
    timeout_seconds: PositiveInt


class VllmProcessSpec(ContractValue):
    """Typed, shell-free launch specification for one vLLM process."""

    process_id: Identifier
    deployment_id: Identifier
    replica_index: NonNegativeInt
    node_group_index: NonNegativeInt
    lane_index: NonNegativeInt
    pipeline_rank: NonNegativeInt
    node_index: NonNegativeInt
    gpu_indices: tuple[NonNegativeInt, ...] = Field(min_length=1)
    role: VllmProcessRole
    tensor_parallel: PositiveInt
    pipeline_parallel: PositiveInt
    http_port: NetworkPort | None = None
    rendezvous: VllmRendezvousSpec | None = None
    launch_delay_seconds: NonNegativeInt

    @model_validator(mode="after")
    def validate_process(self) -> VllmProcessSpec:
        if self.gpu_indices != tuple(sorted(set(self.gpu_indices))):
            raise ValueError("process GPU indices must be sorted and unique")
        if len(self.gpu_indices) != self.tensor_parallel:
            raise ValueError("process GPU count must equal tensor parallelism")
        if self.pipeline_rank >= self.pipeline_parallel:
            raise ValueError("process pipeline rank must be below pipeline parallelism")
        if self.pipeline_rank == 0:
            if self.role is not VllmProcessRole.API_SERVER:
                raise ValueError("pipeline rank zero must be the API server")
            if self.http_port is None:
                raise ValueError("API server processes require an HTTP port")
        elif self.role is not VllmProcessRole.FOLLOWER:
            raise ValueError("nonzero pipeline ranks must be followers")
        elif self.http_port is not None:
            raise ValueError("follower processes must not publish an HTTP port")
        if self.pipeline_parallel == 1:
            if self.pipeline_rank != 0 or self.rendezvous is not None:
                raise ValueError("single-node processes must not carry rendezvous inputs")
        elif self.rendezvous is None:
            raise ValueError("multi-node processes require rendezvous inputs")
        elif self.rendezvous.node_group_index != self.node_group_index or self.rendezvous.lane_index != self.lane_index:
            raise ValueError("process rendezvous identity must match its node group and lane")
        return self
