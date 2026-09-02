# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build every allocation-local process as one structured ``srun`` step."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from data_designer.slurm.config.environment import LiteralEnvironmentBinding, SecretRef
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeEndpoint, RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.serving.deployment import ResolvedVllmServerDeployment
from data_designer.slurm.serving.vllm import ResolvedVllmProcess
from data_designer.slurm.state import AttemptManifest

_SLURM_ENVIRONMENT_NAMES = (
    "SLURM_ARRAY_JOB_ID",
    "SLURM_ARRAY_TASK_ID",
    "SLURM_CLUSTER_NAME",
    "SLURM_CONF",
    "SLURM_CPUS_ON_NODE",
    "SLURM_JOB_CPUS_PER_NODE",
    "SLURM_JOB_GPUS",
    "SLURM_JOB_ID",
    "SLURM_JOB_NODELIST",
    "SLURM_JOB_NUM_NODES",
    "SLURM_NODEID",
    "SLURM_PROCID",
    "SLURM_SUBMIT_DIR",
)


class ClientStepBuilder(Protocol):
    """Construct #876-owned client commands for the common runtime runner."""

    def build_preflight_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        attempt: AttemptManifest,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: Mapping[str, str],
    ) -> RuntimeStep:
        """Build the zero-GPU client preflight step."""
        ...

    def build_generation_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        attempt: AttemptManifest,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: Mapping[str, str],
    ) -> RuntimeStep:
        """Build the zero-GPU client generation step."""
        ...


class DefaultClientStepBuilder:
    """Invoke the allocation client worker supplied by #876."""

    def build_preflight_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        attempt: AttemptManifest,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: Mapping[str, str],
    ) -> RuntimeStep:
        """Build a deterministic client-worker preflight command."""
        return self._build_step(
            "client-preflight",
            RuntimeStepRole.CLIENT_PREFLIGHT,
            "preflight",
            plan,
            shard,
            attempt,
            attempt_directory,
            endpoints,
            source_environment,
        )

    def build_generation_step(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        attempt: AttemptManifest,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: Mapping[str, str],
    ) -> RuntimeStep:
        """Build a deterministic client-worker generation command."""
        return self._build_step(
            "client-generation",
            RuntimeStepRole.CLIENT,
            "run",
            plan,
            shard,
            attempt,
            attempt_directory,
            endpoints,
            source_environment,
        )

    @staticmethod
    def _build_step(
        step_id: str,
        role: RuntimeStepRole,
        operation: str,
        plan: ResolvedSlurmRunPlan,
        shard: PlannedShard,
        attempt: AttemptManifest,
        attempt_directory: Path,
        endpoints: tuple[RuntimeEndpoint, ...],
        source_environment: Mapping[str, str],
    ) -> RuntimeStep:
        endpoint_arguments = tuple(
            argument
            for endpoint in endpoints
            for argument in (
                "--endpoint",
                f"{endpoint.model_alias}=http://{endpoint.host}:{endpoint.port}/v1",
            )
        )
        command = (
            "python3",
            "-m",
            "data_designer.slurm.client.worker",
            operation,
            "--plan",
            get_container_path(plan, plan_path(plan)),
            "--shard-id",
            shard.shard_id,
            "--attempt-id",
            attempt.attempt_id,
            "--attempt-dir",
            get_container_path(plan, attempt_directory.as_posix(), require_writable=True),
            *endpoint_arguments,
        )
        secret_names = _collect_client_secret_environment_names(plan)
        environment = _base_environment(source_environment)
        for name in secret_names:
            try:
                environment[name] = source_environment[name]
            except KeyError:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                    f"required client secret environment {name!r} is unavailable",
                ) from None
        return _build_srun_step(
            step_id=step_id,
            role=role,
            image_path=plan.client.image.path,
            command=command,
            environment=environment,
            container_environment=secret_names,
            plan=plan,
            attempt_directory=attempt_directory,
        )


def build_vllm_steps(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
) -> tuple[RuntimeStep, ...]:
    """Build one structured server step per resolved vLLM process."""
    return tuple(
        _build_vllm_step(deployment, process, plan, attempt_directory, source_environment)
        for process in deployment.processes
    )


def build_endpoint_steps(
    deployments: tuple[ResolvedVllmServerDeployment, ...],
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
    runtime_proxy_path: Path,
) -> tuple[tuple[RuntimeStep, RuntimeEndpoint], ...]:
    """Build one package-owned logical-endpoint process per deployment."""
    steps: list[tuple[RuntimeStep, RuntimeEndpoint]] = []
    for deployment in deployments:
        endpoint = RuntimeEndpoint(
            model_alias=deployment.model_alias,
            served_model_name=deployment.served_model_name,
            host="127.0.0.1",
            port=deployment.logical_endpoint.port,
        )
        backends = tuple(f"http://127.0.0.1:{backend.port}" for backend in deployment.backend_endpoints)
        retry_after_seconds = deployment.launch_policy.queue_backpressure.retry_after_seconds
        retry_arguments = ("--retry-after-seconds", str(retry_after_seconds)) if retry_after_seconds is not None else ()
        command = (
            "python3",
            get_container_path(plan, runtime_proxy_path.as_posix()),
            "--listen-port",
            str(endpoint.port),
            "--max-waiting-requests",
            str(deployment.launch_policy.queue_backpressure.max_waiting_requests),
            *retry_arguments,
            *(argument for backend in backends for argument in ("--backend", backend)),
        )
        step = _build_srun_step(
            step_id=f"{deployment.deployment_id}-endpoint",
            role=RuntimeStepRole.ENDPOINT,
            image_path=plan.client.image.path,
            command=command,
            environment=_base_environment(source_environment),
            container_environment=(),
            plan=plan,
            attempt_directory=attempt_directory,
        )
        steps.append((step, endpoint))
    return tuple(steps)


def plan_path(plan: ResolvedSlurmRunPlan) -> str:
    """Return the canonical persisted resolved-plan path."""
    return str(Path(plan.authored_config.path).with_name("resolved-plan.json"))


def _build_vllm_step(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
) -> RuntimeStep:
    if process.pipeline_parallel != 1 or process.node_index != 0 or process.http_port is None:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "one-node runtime received a distributed vLLM process",
        )
    command: tuple[str, ...] = (
        deployment.executable_path,
        "serve",
        deployment.model,
        "--served-model-name",
        deployment.served_model_name,
        "--host",
        "127.0.0.1",
        "--port",
        str(process.http_port),
        "--tensor-parallel-size",
        str(process.tensor_parallel),
    )
    if deployment.launch_policy.enable_expert_parallel:
        command += ("--enable-expert-parallel",)
    command += deployment.launch_policy.extra_args

    environment = _base_environment(source_environment)
    container_environment = ["CUDA_VISIBLE_DEVICES"]
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(str(index) for index in process.gpu_indices)
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
    return _build_srun_step(
        step_id=process.process_id,
        role=RuntimeStepRole.SERVER,
        image_path=deployment.image.path,
        command=command,
        environment=environment,
        container_environment=tuple(container_environment),
        plan=plan,
        attempt_directory=attempt_directory,
        gpu_indices=tuple(process.gpu_indices),
    )


def _build_srun_step(
    *,
    step_id: str,
    role: RuntimeStepRole,
    image_path: str,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    container_environment: tuple[str, ...],
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    gpu_indices: tuple[int, ...] = (),
) -> RuntimeStep:
    mounts = _render_mounts(plan)
    srun_command = [
        "srun",
        "--nodes=1",
        "--ntasks=1",
        "--exact",
        "--overlap",
        "--unbuffered",
        "--export=ALL",
        f"--cpus-per-task={plan.client.authored.cpus}",
        f"--container-image={image_path}",
    ]
    if gpu_indices:
        rendered_indices = ",".join(str(index) for index in gpu_indices)
        srun_command.extend(
            (
                f"--gpus-per-task={len(gpu_indices)}",
                f"--gpu-bind=map_gpu:{rendered_indices}",
            )
        )
    else:
        srun_command.append("--gres=none")
    if mounts:
        srun_command.append(f"--container-mounts={mounts}")
    if container_environment:
        srun_command.append(f"--container-env={','.join(sorted(set(container_environment)))}")
    srun_command.extend(("--", *command))
    log_root = attempt_directory / "logs"
    return RuntimeStep(
        step_id=step_id,
        role=role,
        command=tuple(srun_command),
        environment=environment,
        stdout_path=log_root / f"{step_id}.out",
        stderr_path=log_root / f"{step_id}.err",
    )


def _render_mounts(plan: ResolvedSlurmRunPlan) -> str:
    rendered: list[str] = []
    for mount in plan.container_mounts:
        if any(character in mount.source or character in mount.target for character in (",", ":")):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.INVALID_CONTEXT,
                "container mount paths cannot contain comma or colon delimiters",
            )
        value = f"{mount.source}:{mount.target}"
        if mount.read_only:
            value += ":ro"
        rendered.append(value)
    return ",".join(rendered)


def _base_environment(source_environment: Mapping[str, str]) -> dict[str, str]:
    environment = {
        "PATH": source_environment.get("PATH") or os.defpath,
        "LC_ALL": "C",
    }
    environment.update(
        (name, source_environment[name]) for name in _SLURM_ENVIRONMENT_NAMES if name in source_environment
    )
    return environment


def _collect_client_secret_environment_names(plan: ResolvedSlurmRunPlan) -> tuple[str, ...]:
    names: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, SecretRef):
            names.add(value.environment)
            return
        if isinstance(value, BaseModel):
            for field_name in type(value).model_fields:
                visit(getattr(value, field_name))
        elif isinstance(value, Mapping):
            for child in value.values():
                visit(child)
        elif isinstance(value, (tuple, list)):
            for child in value:
                visit(child)

    visit(plan.client.authored.dependencies.index_credentials)
    visit(plan.invocation.authored.mcp_providers)
    return tuple(sorted(names))
