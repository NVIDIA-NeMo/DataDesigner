# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed one-node step manifest produced inside the sealed client image."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic import Field, NonNegativeInt, PositiveInt, field_validator, model_validator

from data_designer.slurm.config.environment import (
    LiteralEnvironmentBinding,
    SecretRef,
    collect_secret_environment_names,
)
from data_designer.slurm.contracts import ContractRecord, ContractValue, validate_absolute_path
from data_designer.slurm.runtime.backpressure import (
    MAX_WAITING_REQUESTS_ENVIRONMENT,
    RETRY_AFTER_SECONDS_ENVIRONMENT,
)
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext, RuntimeEndpoint, RuntimeStepRole
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.ports import resolve_allocation_deployments
from data_designer.slurm.runtime.steps import (
    build_client_command,
    build_endpoint_command,
    build_vllm_command,
)
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.vllm import ResolvedVllmProcess
from data_designer.slurm.types import EnvironmentName, Identifier, NetworkPort, Sha256Digest


class RuntimeProbeSpec(ContractValue):
    """One loopback readiness target monitored by the Bash controller."""

    host: Literal["127.0.0.1"] = "127.0.0.1"
    port: NetworkPort
    path: str
    deadline_seconds: PositiveInt

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        if not value.startswith("/") or any(ord(character) < 32 or ord(character) == 127 for character in value):
            raise ValueError("runtime probe path is invalid")
        return value


class RuntimeStepSpec(ContractValue):
    """Container command and environment consumed by the Bash step runner."""

    step_id: Identifier
    role: RuntimeStepRole
    image_path: str
    command: tuple[str, ...] = Field(min_length=1)
    cpus: PositiveInt
    gpu_indices: tuple[NonNegativeInt, ...] = ()
    literal_environment: dict[EnvironmentName, str] = Field(default_factory=dict)
    secret_environment: dict[EnvironmentName, EnvironmentName] = Field(default_factory=dict)
    environment_prefixes: dict[EnvironmentName, str] = Field(default_factory=dict)
    container_environment: tuple[EnvironmentName, ...] = ()
    stdout_path: str
    stderr_path: str
    launch_delay_seconds: NonNegativeInt = 0
    readiness: RuntimeProbeSpec | None = None

    _image_path_is_absolute = field_validator("image_path")(validate_absolute_path)
    _stdout_path_is_absolute = field_validator("stdout_path")(validate_absolute_path)
    _stderr_path_is_absolute = field_validator("stderr_path")(validate_absolute_path)

    @model_validator(mode="after")
    def validate_step(self) -> RuntimeStepSpec:
        if any(not argument or "\0" in argument for argument in self.command):
            raise ValueError("runtime command is invalid")
        if self.gpu_indices != tuple(sorted(set(self.gpu_indices))):
            raise ValueError("runtime GPU indices must be sorted and unique")
        if self.stdout_path == self.stderr_path or Path(self.stdout_path).parent != Path(self.stderr_path).parent:
            raise ValueError("runtime log paths must be distinct siblings")
        if set(self.environment_prefixes) - (set(self.literal_environment) | set(self.secret_environment)):
            raise ValueError("environment prefixes require a materialized variable")
        container_names = set(self.container_environment)
        if container_names - (set(self.literal_environment) | set(self.secret_environment)):
            raise ValueError("container environment contains an unavailable variable")
        if self.role is RuntimeStepRole.SERVER and not self.gpu_indices:
            raise ValueError("server runtime steps require GPUs")
        if self.role is not RuntimeStepRole.SERVER and self.gpu_indices:
            raise ValueError("non-server runtime steps cannot request GPUs")
        return self


class RuntimeBootstrapManifest(ContractRecord):
    """Secret-free one-node allocation command manifest."""

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


def build_runtime_manifest(
    context: AllocationContext,
    *,
    runtime_root: Path,
    log_directory: Path,
) -> RuntimeBootstrapManifest:
    """Build the secret-free one-node command handoff for the Bash controller."""
    plan = context.plan
    runtime_container_root = get_container_path(plan, runtime_root.as_posix(), require_writable=True)
    deployments = resolve_allocation_deployments(context)
    endpoints = tuple(
        RuntimeEndpoint(
            model_alias=deployment.model_alias,
            served_model_name=deployment.served_model_name,
            host="127.0.0.1",
            port=deployment.logical_endpoint.port,
        )
        for deployment in deployments
    )
    steps: list[RuntimeStepSpec] = [
        _build_client_step(
            RuntimeStepRole.CLIENT_PREFLIGHT,
            "client-preflight",
            "preflight",
            context,
            endpoints,
            runtime_container_root,
            log_directory,
        )
    ]
    for deployment in deployments:
        steps.extend(
            _build_server_step(
                deployment,
                process,
                context,
                runtime_root,
                runtime_container_root,
                log_directory,
            )
            for process in deployment.processes
        )
    steps.extend(_build_endpoint_step(deployment, context, runtime_root, log_directory) for deployment in deployments)
    steps.append(
        _build_client_step(
            RuntimeStepRole.CLIENT,
            "client-generation",
            "client",
            context,
            endpoints,
            runtime_container_root,
            log_directory,
        )
    )
    secret_names = set(collect_secret_environment_names(plan))
    secret_names.update(
        name
        for deployment in deployments
        for name, binding in deployment.launch_policy.environment.items()
        if isinstance(binding, SecretRef)
    )
    return RuntimeBootstrapManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=context.shard.shard_id,
        attempt_id=context.attempt.attempt_id,
        plan_sha256=plan.compute_sha256(),
        all_secret_environment_names=tuple(sorted(secret_names)),
        steps=tuple(steps),
    )


def _build_client_step(
    role: RuntimeStepRole,
    step_id: str,
    operation: str,
    context: AllocationContext,
    endpoints: tuple[RuntimeEndpoint, ...],
    runtime_container_root: str,
    log_directory: Path,
) -> RuntimeStepSpec:
    plan = context.plan
    command = build_client_command(
        "preflight" if operation == "preflight" else "run",
        plan,
        context.shard,
        context.attempt,
        context.attempt_directory,
        endpoints,
    )
    if operation == "client":
        command = (
            "python3",
            "-m",
            "data_designer.slurm.runtime.entrypoint",
            "client",
            *command[4:6],
            *command[10:],
        )
    secret_names = collect_secret_environment_names(
        (plan.client.authored.dependencies.index_credentials, plan.invocation.authored.mcp_providers)
    )
    return _step(
        step_id=step_id,
        role=role,
        image_path=plan.client.image.path,
        command=command,
        cpus=plan.client.authored.cpus,
        gpu_indices=(),
        literal_environment={"LC_ALL": "C", "PYTHONPATH": runtime_container_root},
        secret_environment={name: name for name in secret_names},
        environment_prefixes={},
        container_environment=tuple(sorted((*secret_names, "PYTHONPATH"))),
        log_directory=log_directory,
    )


def _build_server_step(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
    context: AllocationContext,
    runtime_root: Path,
    runtime_container_root: str,
    log_directory: Path,
) -> RuntimeStepSpec:
    if process.pipeline_parallel != 1 or process.node_index != 0 or process.http_port is None:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "one-node runtime received a distributed vLLM process",
        )
    literal_environment: dict[str, str] = {"LC_ALL": "C", "PYTHONPATH": runtime_container_root}
    secret_environment: dict[str, str] = {}
    environment_prefixes: dict[str, str] = {}
    for name, binding in deployment.launch_policy.environment.items():
        if isinstance(binding, LiteralEnvironmentBinding):
            literal_environment[name] = binding.value
        elif isinstance(binding, SecretRef):
            secret_environment[name] = binding.environment
        else:  # pragma: no cover - persisted contracts reject unknown bindings
            raise AssertionError(f"unhandled environment binding: {type(binding)!r}")
    if "PYTHONPATH" in secret_environment:
        literal_environment.pop("PYTHONPATH")
        environment_prefixes["PYTHONPATH"] = runtime_container_root
    elif "PYTHONPATH" in deployment.launch_policy.environment:
        literal_environment["PYTHONPATH"] = os.pathsep.join((runtime_container_root, literal_environment["PYTHONPATH"]))
    policy = deployment.launch_policy.queue_backpressure
    literal_environment[MAX_WAITING_REQUESTS_ENVIRONMENT] = str(policy.max_waiting_requests)
    literal_environment[RETRY_AFTER_SECONDS_ENVIRONMENT] = (
        "" if policy.retry_after_seconds is None else str(policy.retry_after_seconds)
    )
    probe = next(item for item in deployment.readiness_probes if item.port == process.http_port)
    return _step(
        step_id=process.process_id,
        role=RuntimeStepRole.SERVER,
        image_path=deployment.image.path,
        command=build_vllm_command(deployment, process, context.plan),
        cpus=context.plan.client.authored.cpus,
        gpu_indices=tuple(process.gpu_indices),
        literal_environment=literal_environment,
        secret_environment=secret_environment,
        environment_prefixes=environment_prefixes,
        container_environment=tuple(
            sorted(
                {
                    *deployment.launch_policy.environment,
                    "PYTHONPATH",
                    MAX_WAITING_REQUESTS_ENVIRONMENT,
                    RETRY_AFTER_SECONDS_ENVIRONMENT,
                }
            )
        ),
        log_directory=log_directory,
        launch_delay_seconds=process.launch_delay_seconds,
        readiness=RuntimeProbeSpec(
            port=probe.port,
            path=probe.path,
            deadline_seconds=probe.deadline_seconds,
        ),
    )


def _build_endpoint_step(
    deployment: ResolvedVllmServerDeployment,
    context: AllocationContext,
    runtime_root: Path,
    log_directory: Path,
) -> RuntimeStepSpec:
    proxy_path = runtime_root / "data_designer/slurm/runtime/proxy.py"
    command = build_endpoint_command(
        deployment,
        context.plan,
        proxy_path,
        deployment.logical_endpoint.port,
    )
    return _step(
        step_id=f"{deployment.deployment_id}-endpoint",
        role=RuntimeStepRole.ENDPOINT,
        image_path=context.plan.client.image.path,
        command=command,
        cpus=context.plan.client.authored.cpus,
        gpu_indices=(),
        literal_environment={"LC_ALL": "C"},
        secret_environment={},
        environment_prefixes={},
        container_environment=(),
        log_directory=log_directory,
        readiness=RuntimeProbeSpec(
            port=deployment.logical_endpoint.port,
            path="/health",
            deadline_seconds=deployment.launch_policy.startup_timeout_seconds,
        ),
    )


def _step(
    *,
    step_id: str,
    role: RuntimeStepRole,
    image_path: str,
    command: tuple[str, ...],
    cpus: int,
    gpu_indices: tuple[int, ...],
    literal_environment: dict[str, str],
    secret_environment: dict[str, str],
    environment_prefixes: dict[str, str],
    container_environment: tuple[str, ...],
    log_directory: Path,
    launch_delay_seconds: int = 0,
    readiness: RuntimeProbeSpec | None = None,
) -> RuntimeStepSpec:
    return RuntimeStepSpec(
        step_id=step_id,
        role=role,
        image_path=image_path,
        command=command,
        cpus=cpus,
        gpu_indices=gpu_indices,
        literal_environment=literal_environment,
        secret_environment=secret_environment,
        environment_prefixes=environment_prefixes,
        container_environment=container_environment,
        stdout_path=(log_directory / f"{step_id}.out").as_posix(),
        stderr_path=(log_directory / f"{step_id}.err").as_posix(),
        launch_delay_seconds=launch_delay_seconds,
        readiness=readiness,
    )


__all__ = ["RuntimeBootstrapManifest", "RuntimeProbeSpec", "RuntimeStepSpec", "build_runtime_manifest"]
