# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose logical endpoint proxies from resolved remote lane heads."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeEndpoint, RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.runtime.step_factory import SrunStepOptions, base_environment, build_srun_step
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment


def build_endpoint_steps(
    deployments: tuple[ResolvedVllmServerDeployment, ...],
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
    runtime_proxy_path: Path,
    layout: AllocationLayout | None = None,
) -> tuple[tuple[RuntimeStep, RuntimeEndpoint], ...]:
    """Build one package-owned logical-endpoint process per deployment."""
    selected_layout = layout or _one_node_layout(plan)
    return tuple(
        _build_endpoint_step(
            deployment,
            plan,
            attempt_directory,
            source_environment,
            runtime_proxy_path,
            selected_layout,
        )
        for deployment in deployments
    )


def _build_endpoint_step(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
    runtime_proxy_path: Path,
    layout: AllocationLayout,
) -> tuple[RuntimeStep, RuntimeEndpoint]:
    endpoint = RuntimeEndpoint(
        model_alias=deployment.model_alias,
        served_model_name=deployment.served_model_name,
        host="127.0.0.1",
        port=deployment.logical_endpoint.port,
    )
    environment = base_environment(source_environment)
    environment["PYTHONPATH"] = get_container_path(plan, runtime_proxy_path.parents[3].as_posix())
    step = build_srun_step(
        step_id=f"{deployment.deployment_id}-endpoint",
        role=RuntimeStepRole.ENDPOINT,
        command=_build_endpoint_command(deployment, plan, runtime_proxy_path, endpoint.port, layout),
        environment=environment,
        plan=plan,
        attempt_directory=attempt_directory,
        options=SrunStepOptions(
            image_path=plan.client.image.path,
            container_environment=("PYTHONPATH",),
            node_hosts=(layout.get_host(plan.client.host_node_index),),
        ),
    )
    return step, endpoint


def _build_endpoint_command(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    runtime_proxy_path: Path,
    port: int,
    layout: AllocationLayout,
) -> tuple[str, ...]:
    backend_hosts = tuple(layout.get_host(backend.node_index) for backend in deployment.backend_endpoints)
    backends = tuple(
        f"http://{host}:{backend.port}"
        for host, backend in zip(backend_hosts, deployment.backend_endpoints, strict=True)
    )
    retry_after_seconds = deployment.launch_policy.queue_backpressure.retry_after_seconds
    retry_arguments = ("--retry-after-seconds", str(retry_after_seconds)) if retry_after_seconds is not None else ()
    return (
        "python3",
        get_container_path(plan, runtime_proxy_path.as_posix()),
        "--listen-port",
        str(port),
        *(argument for host in sorted(set(backend_hosts)) for argument in ("--allowed-host", host)),
        *retry_arguments,
        *(argument for backend in backends for argument in ("--backend", backend)),
    )


def _one_node_layout(plan: ResolvedSlurmRunPlan) -> AllocationLayout:
    node_indices = {
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    }
    if node_indices != {0}:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "multi-node step construction requires a verified allocation layout",
        )
    return AllocationLayout(("127.0.0.1",))


__all__ = ["build_endpoint_steps"]
