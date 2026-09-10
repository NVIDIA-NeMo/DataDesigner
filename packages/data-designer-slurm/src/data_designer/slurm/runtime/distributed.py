# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose resolved deployments into node-worker specifications."""

from __future__ import annotations

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.node_spec import NodeProcessSpec, NodeSpec, NodeWorkerSpec
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.preflight import AllocationLayout
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.vllm import ResolvedVllmProcess


def build_node_worker_spec(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    layout: AllocationLayout,
) -> NodeWorkerSpec:
    """Build the validated work assigned to each node in one deployment."""
    model = _resolve_model(deployment, plan)
    nodes = tuple(
        NodeSpec(
            node_index=node_index,
            host=layout.get_host(node_index),
            ports=_get_node_ports(deployment, node_index),
            processes=tuple(
                NodeProcessSpec(
                    process_id=process.process_id,
                    command=build_vllm_process_command(deployment, process, plan, layout),
                    gpu_indices=tuple(process.gpu_indices),
                    launch_delay_seconds=process.launch_delay_seconds,
                )
                for process in deployment.processes
                if process.node_index == node_index
            ),
        )
        for node_index in deployment.node_indices
    )
    return NodeWorkerSpec(
        schema_version=1,
        resolved_gpus_per_node=deployment.gpus_per_node,
        required_model_path=model if deployment.model.startswith("/") else None,
        nodes=nodes,
    )


def build_vllm_process_command(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
    plan: ResolvedSlurmRunPlan,
    layout: AllocationLayout,
) -> tuple[str, ...]:
    """Build one shell-free vLLM lane command at its resolved host placement."""
    backend = deployment.backend_endpoints[process.deployment_replica_index]
    model = _resolve_model(deployment, plan)
    command: tuple[str, ...] = (
        deployment.executable_path,
        "serve",
        model,
        "--served-model-name",
        deployment.served_model_name,
        "--host",
        "0.0.0.0",
        "--port",
        str(backend.port),
        "--tensor-parallel-size",
        str(process.tensor_parallel),
        "--distributed-executor-backend",
        "uni" if process.tensor_parallel * process.pipeline_parallel == 1 else "mp",
        "--data-parallel-backend",
        "mp",
        "--middleware",
        "data_designer.slurm.runtime.backpressure.QueueDepthBackpressureMiddleware",
        *_distributed_arguments(process, layout),
    )
    if deployment.launch_policy.enable_expert_parallel:
        command += ("--enable-expert-parallel",)
    return command + deployment.launch_policy.extra_args


def _resolve_model(deployment: ResolvedVllmServerDeployment, plan: ResolvedSlurmRunPlan) -> str:
    return get_container_path(plan, deployment.model) if deployment.model.startswith("/") else deployment.model


def _distributed_arguments(
    process: ResolvedVllmProcess,
    layout: AllocationLayout,
) -> tuple[str, ...]:
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


def _get_node_ports(deployment: ResolvedVllmServerDeployment, node_index: int) -> tuple[int, ...]:
    http_ports = tuple(endpoint.port for endpoint in deployment.backend_endpoints if endpoint.node_index == node_index)
    rendezvous_ports = tuple(
        process.rendezvous.port
        for process in deployment.processes
        if process.node_index == node_index and process.pipeline_rank == 0 and process.rendezvous is not None
    )
    return tuple(sorted((*http_ports, *rendezvous_ports)))


__all__ = ["build_node_worker_spec", "build_vllm_process_command"]
