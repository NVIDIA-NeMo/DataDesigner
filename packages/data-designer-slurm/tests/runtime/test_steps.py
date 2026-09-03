# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from conftest import RuntimeCase, relocate_plan

from data_designer.slurm.config import QueueBackpressureConfig
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import RuntimeEndpoint, RuntimeStepRole
from data_designer.slurm.runtime.steps import (
    DefaultClientStepBuilder,
    build_endpoint_steps,
    build_vllm_steps,
)
from data_designer.slurm.serving.resolver import resolve_vllm_server
from data_designer.slurm.state import AttemptLifecycleState, AttemptManifest, SchedulerIdentity


def test_all_processes_use_structured_srun_steps_and_sanitized_environment(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    deployment = resolve_vllm_server(context.plan, context.plan.deployments[0].deployment_id)
    source_environment = {
        "PATH": "/custom/bin",
        "SLURM_JOB_ID": "4101",
        "UNREVIEWED_VALUE": "must-not-forward",
    }

    server_steps = build_vllm_steps(
        deployment,
        context.plan,
        context.attempt_directory,
        source_environment,
    )
    endpoint_steps = build_endpoint_steps(
        (deployment,),
        context.plan,
        context.attempt_directory,
        source_environment,
        context.attempt_directory / "runtime/proxy.py",
    )
    endpoints = tuple(endpoint for _, endpoint in endpoint_steps)
    client_builder = DefaultClientStepBuilder()
    client_steps = (
        client_builder.build_preflight_step(
            context.plan,
            context.shard,
            context.attempt,
            context.attempt_directory,
            endpoints,
            source_environment,
        ),
        client_builder.build_generation_step(
            context.plan,
            context.shard,
            context.attempt,
            context.attempt_directory,
            endpoints,
            source_environment,
        ),
    )

    all_steps = server_steps + tuple(step for step, _ in endpoint_steps) + client_steps
    assert all(step.command[0] == "srun" for step in all_steps)
    assert all("--" in step.command for step in all_steps)
    assert all("UNREVIEWED_VALUE" not in step.environment for step in all_steps)
    assert all(step.environment["SLURM_JOB_ID"] == "4101" for step in all_steps)
    assert all(step.environment["LC_ALL"] == "C" for step in all_steps)
    assert all(step.stdout_path.parent == context.attempt_directory / "logs" for step in all_steps)
    assert all(f"--cpus-per-task={context.plan.client.authored.cpus}" in step.command for step in all_steps)

    server = server_steps[0]
    assert server.role is RuntimeStepRole.SERVER
    assert f"--gpus-per-task={deployment.topology.tensor_parallel}" in server.command
    expected_mask = sum(1 << index for index in deployment.processes[0].gpu_indices)
    assert f"--gpu-bind=mask_gpu:{expected_mask:#x}" in server.command
    assert "CUDA_VISIBLE_DEVICES" not in server.environment
    assert all("CUDA_VISIBLE_DEVICES" not in argument for argument in server.command)
    assert all("--gpus-per-task=" not in argument for step in client_steps for argument in step.command)
    assert all("--gres=none" in step.command for step in client_steps)
    assert all("CUDA_VISIBLE_DEVICES" not in step.environment for step in client_steps)


def test_client_worker_receives_only_persisted_identity_and_logical_endpoint(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    deployment = resolve_vllm_server(context.plan, context.plan.deployments[0].deployment_id)
    endpoint_steps = build_endpoint_steps(
        (deployment,),
        context.plan,
        context.attempt_directory,
        {},
        context.attempt_directory / "runtime/proxy.py",
    )
    endpoints = tuple(endpoint for _, endpoint in endpoint_steps)

    step = DefaultClientStepBuilder().build_generation_step(
        context.plan,
        context.shard,
        context.attempt,
        context.attempt_directory,
        endpoints,
        {},
    )

    separator = step.command.index("--")
    worker = step.command[separator + 1 :]
    assert worker[:5] == (
        "python3",
        "-m",
        "data_designer.slurm.client.worker",
        "run",
        "--plan",
    )
    assert context.attempt.attempt_id in worker
    assert context.shard.shard_id in worker
    assert f"generator=http://127.0.0.1:{endpoints[0].port}/v1" in worker


def test_endpoint_step_uses_resolved_retry_policy_and_backends(runtime_case: RuntimeCase) -> None:
    context = runtime_case.context
    deployment = resolve_vllm_server(context.plan, context.plan.deployments[0].deployment_id)

    ((step, endpoint),) = build_endpoint_steps(
        (deployment,),
        context.plan,
        context.attempt_directory,
        {},
        context.attempt_directory / "runtime/proxy.py",
    )

    assert step.role is RuntimeStepRole.ENDPOINT
    assert endpoint.port == deployment.logical_endpoint.port
    retry_after_seconds = deployment.launch_policy.queue_backpressure.retry_after_seconds
    assert retry_after_seconds is not None
    assert ("--retry-after-seconds", str(retry_after_seconds)) == step.command[
        step.command.index("--retry-after-seconds") : step.command.index("--retry-after-seconds") + 2
    ]
    assert "--max-waiting-requests" not in step.command
    for backend in deployment.backend_endpoints:
        assert f"http://127.0.0.1:{backend.port}" in step.command

    disabled_policy = deployment.launch_policy.model_copy(
        update={"queue_backpressure": QueueBackpressureConfig(max_waiting_requests=0, retry_after_seconds=None)}
    )
    disabled = deployment.model_copy(update={"launch_policy": disabled_policy})
    ((disabled_step, _),) = build_endpoint_steps(
        (disabled,),
        context.plan,
        context.attempt_directory,
        {},
        context.attempt_directory / "runtime/proxy.py",
    )
    assert "--retry-after-seconds" not in disabled_step.command
    assert "--max-waiting-requests" not in disabled_step.command


def test_client_receives_client_secrets_without_server_only_secrets(
    tmp_path: Path,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    plan = relocate_plan(multi_node_plan, workspace)
    shard = plan.shards[0]
    attempt_directory = workspace / "attempt"
    attempt = AttemptManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=ArtifactReference(
            path=Path(plan.authored_config.path).with_name("resolved-plan.json").as_posix(),
            sha256=plan.compute_sha256(),
        ),
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0),
        created_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        updated_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
    )
    endpoints = tuple(
        RuntimeEndpoint(
            model_alias=deployment.authored.model_alias,
            served_model_name=deployment.served_model_name,
            host="127.0.0.1",
            port=port.port,
        )
        for deployment, port in zip(plan.deployments, plan.client.ports, strict=True)
    )

    step = DefaultClientStepBuilder().build_preflight_step(
        plan,
        shard,
        attempt,
        attempt_directory,
        endpoints,
        {
            "PACKAGE_INDEX_TOKEN": "client-secret",
            "HF_TOKEN": "server-secret",
        },
    )

    assert step.environment["PACKAGE_INDEX_TOKEN"] == "client-secret"
    assert "HF_TOKEN" not in step.environment
    assert "--container-env=PACKAGE_INDEX_TOKEN" in step.command

    with pytest.raises(SlurmRuntimeError, match="PACKAGE_INDEX_TOKEN"):
        DefaultClientStepBuilder().build_preflight_step(
            plan,
            shard,
            attempt,
            attempt_directory,
            endpoints,
            {"HF_TOKEN": "server-secret"},
        )
