# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed allocation step manifest produced inside the sealed client image."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from data_designer.slurm.config.environment import (
    SecretRef,
    collect_secret_environment_names,
)
from data_designer.slurm.runtime.manifest import RuntimeBootstrapManifest, RuntimeProbeSpec, RuntimeStepSpec
from data_designer.slurm.runtime.models import AllocationContext, RuntimeEndpoint, RuntimeStepRole
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.ports import resolve_allocation_deployments
from data_designer.slurm.runtime.preflight import AllocationLayout, validate_allocation_layout
from data_designer.slurm.runtime.server_manifest import build_server_steps
from data_designer.slurm.runtime.steps import build_client_command, build_endpoint_command
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment


def build_runtime_manifest(
    context: AllocationContext,
    environment: Mapping[str, str],
    *,
    runtime_root: Path,
    log_directory: Path,
    layout: AllocationLayout,
) -> RuntimeBootstrapManifest:
    """Build the secret-free command handoff for the Bash controller."""
    plan = context.plan
    validate_allocation_layout(plan, layout)
    runtime_container_root = get_container_path(plan, runtime_root.as_posix(), require_writable=True)
    deployments = resolve_allocation_deployments(context, environment)
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
            environment,
            endpoints,
            runtime_container_root,
            log_directory,
            layout,
        )
    ]
    steps.extend(build_server_steps(deployments, context, runtime_container_root, log_directory, layout))
    steps.extend(
        _build_endpoint_step(deployment, context, runtime_root, log_directory, layout) for deployment in deployments
    )
    steps.append(
        _build_client_step(
            RuntimeStepRole.CLIENT,
            "client-generation",
            "client",
            context,
            environment,
            endpoints,
            runtime_container_root,
            log_directory,
            layout,
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
    environment: Mapping[str, str],
    endpoints: tuple[RuntimeEndpoint, ...],
    runtime_container_root: str,
    log_directory: Path,
    layout: AllocationLayout,
) -> RuntimeStepSpec:
    plan = context.plan
    retry_resume_mode = None if context.retry_plan is None else context.retry_plan.effective_resume_mode
    command = build_client_command(
        "preflight" if operation == "preflight" else "run",
        plan,
        context.shard,
        context.attempt,
        context.attempt_directory,
        endpoints,
        retry_resume_mode,
    )
    if operation == "client":
        endpoint_arguments = tuple(
            argument
            for endpoint in endpoints
            for argument in ("--endpoint", f"{endpoint.model_alias}=http://{endpoint.host}:{endpoint.port}/v1")
        )
        retry_binding = (
            ()
            if context.retry_plan is None
            else (
                "--retry-id",
                context.retry_plan.retry_id,
                "--retry-plan-sha256",
                context.retry_plan.compute_sha256(),
                "--effective-resume-mode",
                context.retry_plan.effective_resume_mode,
            )
        )
        command = (
            "python3",
            "-m",
            "data_designer.slurm.runtime.entrypoint",
            "client",
            *command[4:6],
            *command[10:12],
            *retry_binding,
            *endpoint_arguments,
        )
    secret_names = collect_secret_environment_names(
        (plan.client.authored.dependencies.index_credentials, plan.invocation.authored.mcp_providers)
    )
    allocation_environment = (
        {"SLURM_JOB_GPUS": environment["SLURM_JOB_GPUS"]}
        if plan.selected_profile.profile.gpu_request_mode == "gres"
        else {}
    )
    return _step(
        step_id=step_id,
        role=role,
        image_path=plan.client.image.path,
        command=command,
        cpus=plan.client.authored.cpus,
        gpu_indices=(),
        literal_environment={"LC_ALL": "C", "PYTHONPATH": runtime_container_root, **allocation_environment},
        secret_environment={name: name for name in secret_names},
        environment_prefixes={},
        container_environment=tuple(sorted((*secret_names, *allocation_environment, "PYTHONPATH"))),
        log_directory=log_directory,
        node_hosts=(layout.get_host(plan.client.host_node_index),),
    )


def _build_endpoint_step(
    deployment: ResolvedVllmServerDeployment,
    context: AllocationContext,
    runtime_root: Path,
    log_directory: Path,
    layout: AllocationLayout,
) -> RuntimeStepSpec:
    proxy_path = runtime_root / "data_designer/slurm/runtime/proxy.py"
    backend_hosts = (
        tuple(layout.get_host(backend.node_index) for backend in deployment.backend_endpoints)
        if len(layout.node_hosts) > 1
        else None
    )
    command = build_endpoint_command(
        deployment,
        context.plan,
        proxy_path,
        deployment.logical_endpoint.port,
        backend_hosts=backend_hosts,
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
        node_hosts=(layout.get_host(context.plan.client.host_node_index),),
        readiness=(
            RuntimeProbeSpec(
                host="127.0.0.1",
                port=deployment.logical_endpoint.port,
                path="/health",
                deadline_seconds=deployment.launch_policy.startup_timeout_seconds,
            ),
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
    node_hosts: tuple[str, ...],
    literal_environment: dict[str, str],
    secret_environment: dict[str, str],
    environment_prefixes: dict[str, str],
    container_environment: tuple[str, ...],
    log_directory: Path,
    kill_on_bad_exit: bool = False,
    launch_delay_seconds: int = 0,
    readiness: tuple[RuntimeProbeSpec, ...] = (),
) -> RuntimeStepSpec:
    return RuntimeStepSpec(
        step_id=step_id,
        role=role,
        image_path=image_path,
        command=command,
        cpus=cpus,
        gpu_indices=gpu_indices,
        node_hosts=node_hosts,
        kill_on_bad_exit=kill_on_bad_exit,
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
