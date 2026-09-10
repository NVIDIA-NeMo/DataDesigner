# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose local and distributed server entries for the runtime manifest."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from data_designer.slurm.config.environment import LiteralEnvironmentBinding, SecretRef
from data_designer.slurm.runtime.backpressure import (
    MAX_WAITING_REQUESTS_ENVIRONMENT,
    RETRY_AFTER_SECONDS_ENVIRONMENT,
)
from data_designer.slurm.runtime.distributed import build_node_worker_spec, build_vllm_process_command
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.manifest import RuntimeProbeSpec, RuntimeStepSpec
from data_designer.slurm.runtime.models import AllocationContext, RuntimeStepRole
from data_designer.slurm.runtime.node_spec import encode_node_worker_spec
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.runtime.steps import build_vllm_command
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.vllm import ResolvedVllmProcess


def build_server_steps(
    deployments: tuple[ResolvedVllmServerDeployment, ...],
    context: AllocationContext,
    runtime_container_root: str,
    log_directory: Path,
    layout: AllocationLayout,
) -> tuple[RuntimeStepSpec, ...]:
    """Build remote preflight entries followed by all serving entries."""
    preflight_steps = tuple(
        build_distributed_server_step(
            deployment,
            context,
            runtime_container_root,
            log_directory,
            layout,
            operation="preflight",
        )
        for deployment in deployments
        if _requires_server_preflight(deployment, context)
    )
    serving_steps = tuple(
        step
        for deployment in deployments
        for step in _build_deployment_serving_steps(
            deployment,
            context,
            runtime_container_root,
            log_directory,
            layout,
        )
    )
    return preflight_steps + serving_steps


def build_distributed_server_step(
    deployment: ResolvedVllmServerDeployment,
    context: AllocationContext,
    runtime_container_root: str,
    log_directory: Path,
    layout: AllocationLayout,
    *,
    operation: Literal["preflight", "serve"],
) -> RuntimeStepSpec:
    """Build one coordinated node-worker step for a deployment."""
    literal_environment, secret_environment, environment_prefixes = _server_environment(
        deployment,
        runtime_container_root,
    )
    worker_spec = build_node_worker_spec(deployment, context.plan, layout)
    readiness = (
        tuple(
            RuntimeProbeSpec(
                host=layout.get_host(probe.node_index),
                port=probe.port,
                path=probe.path,
                deadline_seconds=probe.deadline_seconds,
            )
            for probe in deployment.readiness_probes
        )
        if operation == "serve"
        else ()
    )
    step_id = f"{deployment.deployment_id}-{operation}"
    return RuntimeStepSpec(
        step_id=step_id,
        role=RuntimeStepRole.SERVER if operation == "serve" else RuntimeStepRole.SERVER_PREFLIGHT,
        image_path=deployment.image.path,
        command=(
            "python3",
            f"{runtime_container_root}/data_designer/slurm/runtime/node_worker.py",
            operation,
            "--spec",
            encode_node_worker_spec(worker_spec),
        ),
        cpus=context.plan.client.authored.cpus,
        gpu_indices=tuple(range(deployment.gpus_per_node)),
        node_hosts=tuple(layout.get_host(index) for index in deployment.node_indices),
        kill_on_bad_exit=True,
        literal_environment=literal_environment,
        secret_environment=secret_environment,
        environment_prefixes=environment_prefixes,
        container_environment=_server_container_environment(deployment),
        stdout_path=(log_directory / f"{step_id}.out").as_posix(),
        stderr_path=(log_directory / f"{step_id}.err").as_posix(),
        readiness=readiness,
    )


def _build_deployment_serving_steps(
    deployment: ResolvedVllmServerDeployment,
    context: AllocationContext,
    runtime_container_root: str,
    log_directory: Path,
    layout: AllocationLayout,
) -> tuple[RuntimeStepSpec, ...]:
    if len(deployment.node_indices) > 1:
        return (
            build_distributed_server_step(
                deployment,
                context,
                runtime_container_root,
                log_directory,
                layout,
                operation="serve",
            ),
        )
    return tuple(
        _build_local_server_step(
            deployment,
            process,
            context,
            runtime_container_root,
            log_directory,
            layout,
        )
        for process in deployment.processes
    )


def _build_local_server_step(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
    context: AllocationContext,
    runtime_container_root: str,
    log_directory: Path,
    layout: AllocationLayout,
) -> RuntimeStepSpec:
    if process.pipeline_parallel != 1 or process.http_port is None:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "single-node runtime received a distributed vLLM process",
        )
    literal_environment, secret_environment, environment_prefixes = _server_environment(
        deployment,
        runtime_container_root,
    )
    probe = next(item for item in deployment.readiness_probes if item.port == process.http_port)
    probe_host = "127.0.0.1" if len(layout.node_hosts) == 1 else layout.get_host(process.node_index)
    command = (
        build_vllm_command(deployment, process, context.plan)
        if len(layout.node_hosts) == 1
        else build_vllm_process_command(deployment, process, context.plan, layout)
    )
    return RuntimeStepSpec(
        step_id=process.process_id,
        role=RuntimeStepRole.SERVER,
        image_path=deployment.image.path,
        command=command,
        cpus=context.plan.client.authored.cpus,
        gpu_indices=tuple(process.gpu_indices),
        node_hosts=(layout.get_host(process.node_index),),
        literal_environment=literal_environment,
        secret_environment=secret_environment,
        environment_prefixes=environment_prefixes,
        container_environment=_server_container_environment(deployment),
        stdout_path=(log_directory / f"{process.process_id}.out").as_posix(),
        stderr_path=(log_directory / f"{process.process_id}.err").as_posix(),
        launch_delay_seconds=process.launch_delay_seconds,
        readiness=(
            RuntimeProbeSpec(
                host=probe_host,
                port=probe.port,
                path=probe.path,
                deadline_seconds=probe.deadline_seconds,
            ),
        ),
    )


def _server_environment(
    deployment: ResolvedVllmServerDeployment,
    runtime_container_root: str,
) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
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
    return literal_environment, secret_environment, environment_prefixes


def _server_container_environment(deployment: ResolvedVllmServerDeployment) -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                *deployment.launch_policy.environment,
                "PYTHONPATH",
                MAX_WAITING_REQUESTS_ENVIRONMENT,
                RETRY_AFTER_SECONDS_ENVIRONMENT,
            }
        )
    )


def _requires_server_preflight(
    deployment: ResolvedVllmServerDeployment,
    context: AllocationContext,
) -> bool:
    return any(node_index != context.plan.client.host_node_index for node_index in deployment.node_indices)


__all__ = ["build_distributed_server_step", "build_server_steps"]
