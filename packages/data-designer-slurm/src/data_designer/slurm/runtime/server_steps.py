# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose resolved vLLM deployments into coordinated multi-node steps."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from data_designer.slurm.config.environment import LiteralEnvironmentBinding, SecretRef
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.backpressure import (
    MAX_WAITING_REQUESTS_ENVIRONMENT,
    RETRY_AFTER_SECONDS_ENVIRONMENT,
)
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.node_spec import NodeProcessSpec, NodeSpec, NodeWorkerSpec, encode_node_worker_spec
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.runtime.step_factory import SrunStepOptions, base_environment, build_srun_step
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.vllm import ResolvedVllmProcess


def build_vllm_steps(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
    runtime_node_worker_path: Path | None = None,
    layout: AllocationLayout | None = None,
    *,
    preflight: bool = False,
) -> tuple[RuntimeStep, ...]:
    """Build one coordinated multi-node step for a resolved vLLM deployment."""
    selected_layout = layout or _one_node_layout(plan)
    worker_path = runtime_node_worker_path or attempt_directory / "runtime/data_designer/slurm/runtime/node_worker.py"
    worker_spec = _build_node_worker_spec(deployment, selected_layout)
    operation = "preflight" if preflight else "serve"
    role = RuntimeStepRole.SERVER_PREFLIGHT if preflight else RuntimeStepRole.SERVER
    environment, container_environment = _build_vllm_environment(
        deployment,
        source_environment,
        plan,
        worker_path,
    )
    step = build_srun_step(
        step_id=f"{deployment.deployment_id}-{operation}",
        role=role,
        command=(
            "python3",
            get_container_path(plan, worker_path.as_posix()),
            operation,
            "--spec",
            encode_node_worker_spec(worker_spec),
        ),
        environment=environment,
        plan=plan,
        attempt_directory=attempt_directory,
        options=SrunStepOptions(
            image_path=deployment.image.path,
            container_environment=container_environment,
            node_hosts=tuple(selected_layout.get_host(index) for index in deployment.node_indices),
            gpu_count=deployment.gpus_per_node,
            kill_on_bad_exit=True,
        ),
    )
    return (step,)


def _build_node_worker_spec(
    deployment: ResolvedVllmServerDeployment,
    layout: AllocationLayout,
) -> NodeWorkerSpec:
    nodes = tuple(
        NodeSpec(
            node_index=node_index,
            host=layout.get_host(node_index),
            ports=_get_node_ports(deployment, node_index),
            processes=tuple(
                NodeProcessSpec(
                    process_id=process.process_id,
                    command=_build_vllm_command(deployment, process, layout),
                    gpu_indices=tuple(process.gpu_indices),
                    launch_delay_seconds=process.launch_delay_seconds,
                )
                for process in deployment.processes
                if process.node_index == node_index
            ),
        )
        for node_index in deployment.node_indices
    )
    return NodeWorkerSpec(schema_version=1, resolved_gpus_per_node=deployment.gpus_per_node, nodes=nodes)


def _get_node_ports(deployment: ResolvedVllmServerDeployment, node_index: int) -> tuple[int, ...]:
    http_ports = tuple(endpoint.port for endpoint in deployment.backend_endpoints if endpoint.node_index == node_index)
    rendezvous_ports = tuple(
        process.rendezvous.port
        for process in deployment.processes
        if process.node_index == node_index and process.pipeline_rank == 0 and process.rendezvous is not None
    )
    return tuple(sorted((*http_ports, *rendezvous_ports)))


def _build_vllm_command(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
    layout: AllocationLayout,
) -> tuple[str, ...]:
    backend = deployment.backend_endpoints[process.deployment_replica_index]
    command: tuple[str, ...] = (
        deployment.executable_path,
        "serve",
        deployment.model,
        "--served-model-name",
        deployment.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(backend.port),
        "--tensor-parallel-size",
        str(process.tensor_parallel),
        "--distributed-executor-backend",
        "uni" if process.tensor_parallel == 1 else "mp",
        "--data-parallel-backend",
        "mp",
        "--middleware",
        "data_designer.slurm.runtime.backpressure.QueueDepthBackpressureMiddleware",
    )
    command += _distributed_arguments(process, layout)
    if deployment.launch_policy.enable_expert_parallel:
        command += ("--enable-expert-parallel",)
    return command + deployment.launch_policy.extra_args


def _distributed_arguments(process: ResolvedVllmProcess, layout: AllocationLayout) -> tuple[str, ...]:
    if process.pipeline_parallel == 1:
        return ()
    rendezvous = process.rendezvous
    if rendezvous is None:  # pragma: no cover - resolved contracts enforce this
        raise AssertionError("distributed process has no rendezvous")
    arguments = (
        "--pipeline-parallel-size",
        str(process.pipeline_parallel),
        "--nnodes",
        str(process.pipeline_parallel),
        "--node-rank",
        str(process.pipeline_rank),
        "--master-addr",
        layout.get_host(rendezvous.master_node_index),
        "--master-port",
        str(rendezvous.port),
        "--distributed-timeout-seconds",
        str(rendezvous.timeout_seconds),
    )
    return arguments + (("--headless",) if process.pipeline_rank > 0 else ())


def _build_vllm_environment(
    deployment: ResolvedVllmServerDeployment,
    source_environment: Mapping[str, str],
    plan: ResolvedSlurmRunPlan,
    runtime_node_worker_path: Path,
) -> tuple[dict[str, str], tuple[str, ...]]:
    environment = base_environment(source_environment)
    container_environment: list[str] = []
    for name, binding in deployment.launch_policy.environment.items():
        if isinstance(binding, LiteralEnvironmentBinding):
            environment[name] = binding.value
        elif isinstance(binding, SecretRef):
            try:
                environment[name] = source_environment[binding.environment]
            except KeyError:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                    f"required server secret environment {binding.environment!r} is unavailable",
                ) from None
        else:  # pragma: no cover - persisted contracts reject unknown bindings
            raise AssertionError(f"unhandled environment binding: {type(binding)!r}")
        container_environment.append(name)
    runtime_root = runtime_node_worker_path.parents[3]
    environment["PYTHONPATH"] = get_container_path(plan, runtime_root.as_posix())
    queue_policy = deployment.launch_policy.queue_backpressure
    environment[MAX_WAITING_REQUESTS_ENVIRONMENT] = str(queue_policy.max_waiting_requests)
    environment[RETRY_AFTER_SECONDS_ENVIRONMENT] = (
        "" if queue_policy.retry_after_seconds is None else str(queue_policy.retry_after_seconds)
    )
    container_environment.extend(("PYTHONPATH", MAX_WAITING_REQUESTS_ENVIRONMENT, RETRY_AFTER_SECONDS_ENVIRONMENT))
    return environment, tuple(container_environment)


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


__all__ = ["build_vllm_steps"]
