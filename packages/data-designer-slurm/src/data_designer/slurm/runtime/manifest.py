# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated command manifest consumed by the allocation shell runtime."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.contracts import ContractRecord, ContractValue, validate_absolute_path
from data_designer.slurm.runtime.models import RuntimeStepRole
from data_designer.slurm.runtime.network import validate_host_name
from data_designer.slurm.types import EnvironmentName, Identifier, NetworkPort, Sha256Digest


class RuntimeProbeSpec(ContractValue):
    """One readiness target monitored from the allocation client host."""

    host: str
    port: NetworkPort
    path: str
    deadline_seconds: PositiveInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.startswith("/") or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("runtime probe path is invalid")
        return value

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        return validate_host_name(value)


class RuntimeStepSpec(ContractValue):
    """Container command and placement consumed by the Bash step runner."""

    step_id: Identifier
    role: RuntimeStepRole
    image_path: str
    command: tuple[str, ...] = Field(min_length=1)
    cpus: PositiveInt
    gpu_indices: tuple[NonNegativeInt, ...] = ()
    node_hosts: tuple[str, ...] = Field(min_length=1)
    kill_on_bad_exit: bool = False
    literal_environment: dict[EnvironmentName, str] = Field(default_factory=dict)
    secret_environment: dict[EnvironmentName, EnvironmentName] = Field(default_factory=dict)
    environment_prefixes: dict[EnvironmentName, str] = Field(default_factory=dict)
    container_environment: tuple[EnvironmentName, ...] = ()
    stdout_path: str
    stderr_path: str
    launch_delay_seconds: NonNegativeInt = 0
    readiness: tuple[RuntimeProbeSpec, ...] = ()

    _image_path_is_absolute = field_validator("image_path")(validate_absolute_path)
    _stdout_path_is_absolute = field_validator("stdout_path")(validate_absolute_path)
    _stderr_path_is_absolute = field_validator("stderr_path")(validate_absolute_path)

    @model_validator(mode="after")
    def validate_step(self) -> RuntimeStepSpec:
        if any(not argument or "\0" in argument for argument in self.command):
            raise ValueError("runtime command is invalid")
        if self.gpu_indices != tuple(sorted(set(self.gpu_indices))):
            raise ValueError("runtime GPU indices must be sorted and unique")
        if self.node_hosts != tuple(dict.fromkeys(self.node_hosts)):
            raise ValueError("runtime node hosts must be unique")
        for host in self.node_hosts:
            validate_host_name(host)
        if self.stdout_path == self.stderr_path or Path(self.stdout_path).parent != Path(self.stderr_path).parent:
            raise ValueError("runtime log paths must be distinct siblings")
        if set(self.environment_prefixes) - (set(self.literal_environment) | set(self.secret_environment)):
            raise ValueError("environment prefixes require a materialized variable")
        container_names = set(self.container_environment)
        if container_names - (set(self.literal_environment) | set(self.secret_environment)):
            raise ValueError("container environment contains an unavailable variable")
        self._validate_placement()
        self._validate_readiness()
        return self

    def _validate_placement(self) -> None:
        server_roles = {RuntimeStepRole.SERVER_PREFLIGHT, RuntimeStepRole.SERVER}
        if self.role in server_roles and not self.gpu_indices:
            raise ValueError("server runtime steps require GPUs")
        if self.role not in server_roles and self.gpu_indices:
            raise ValueError("non-server runtime steps cannot request GPUs")
        if len(self.node_hosts) > 1 and self.role not in server_roles:
            raise ValueError("only server runtime steps may span nodes")
        if self.kill_on_bad_exit and self.role not in server_roles:
            raise ValueError("kill-on-bad-exit is only valid for server runtime steps")
        if len(self.node_hosts) > 1 and not self.kill_on_bad_exit:
            raise ValueError("multi-node server runtime steps require kill-on-bad-exit")

    def _validate_readiness(self) -> None:
        if self.role is RuntimeStepRole.SERVER_PREFLIGHT and self.readiness:
            raise ValueError("server preflight steps cannot have readiness probes")
        if self.role in {RuntimeStepRole.SERVER, RuntimeStepRole.ENDPOINT} and not self.readiness:
            raise ValueError("long-running runtime steps require readiness probes")
        if self.role in {RuntimeStepRole.CLIENT_PREFLIGHT, RuntimeStepRole.CLIENT} and self.readiness:
            raise ValueError("client runtime steps cannot have readiness probes")


class RuntimeBootstrapManifest(ContractRecord):
    """Secret-free allocation command manifest."""

    run_id: Identifier
    shard_id: Identifier
    attempt_id: Identifier
    plan_sha256: Sha256Digest
    all_secret_environment_names: tuple[EnvironmentName, ...]
    steps: tuple[RuntimeStepSpec, ...] = Field(min_length=4)

    @model_validator(mode="after")
    def validate_steps(self) -> RuntimeBootstrapManifest:
        step_ids = tuple(step.step_id for step in self.steps)
        if len(step_ids) != len(set(step_ids)):
            raise ValueError("runtime step identifiers must be unique")
        roles = tuple(step.role for step in self.steps)
        if roles.count(RuntimeStepRole.CLIENT_PREFLIGHT) != 1 or roles.count(RuntimeStepRole.CLIENT) != 1:
            raise ValueError("runtime manifest requires one preflight and generation step")
        if RuntimeStepRole.SERVER not in roles or RuntimeStepRole.ENDPOINT not in roles:
            raise ValueError("runtime manifest requires server and endpoint steps")
        return self


__all__ = ["RuntimeBootstrapManifest", "RuntimeProbeSpec", "RuntimeStepSpec"]
