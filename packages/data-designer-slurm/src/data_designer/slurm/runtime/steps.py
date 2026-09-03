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
        command = _build_client_command(operation, plan, shard, attempt, attempt_directory, endpoints)
        secret_names, environment = _build_client_environment(plan, source_environment)
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


def _build_client_command(
    operation: str,
    plan: ResolvedSlurmRunPlan,
    shard: PlannedShard,
    attempt: AttemptManifest,
    attempt_directory: Path,
    endpoints: tuple[RuntimeEndpoint, ...],
) -> tuple[str, ...]:
    endpoint_arguments = tuple(
        argument
        for endpoint in endpoints
        for argument in ("--endpoint", f"{endpoint.model_alias}=http://{endpoint.host}:{endpoint.port}/v1")
    )
    return (
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


def _build_client_environment(
    plan: ResolvedSlurmRunPlan,
    source_environment: Mapping[str, str],
) -> tuple[tuple[str, ...], dict[str, str]]:
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
    return secret_names, environment


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
        steps.append(_build_endpoint_step(deployment, plan, attempt_directory, source_environment, runtime_proxy_path))
    return tuple(steps)


def _build_endpoint_step(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    source_environment: Mapping[str, str],
    runtime_proxy_path: Path,
) -> tuple[RuntimeStep, RuntimeEndpoint]:
    endpoint = RuntimeEndpoint(
        model_alias=deployment.model_alias,
        served_model_name=deployment.served_model_name,
        host="127.0.0.1",
        port=deployment.logical_endpoint.port,
    )
    command = _build_endpoint_command(deployment, plan, runtime_proxy_path, endpoint.port)
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
    return step, endpoint


def _build_endpoint_command(
    deployment: ResolvedVllmServerDeployment,
    plan: ResolvedSlurmRunPlan,
    runtime_proxy_path: Path,
    port: int,
) -> tuple[str, ...]:
    backends = tuple(f"http://127.0.0.1:{backend.port}" for backend in deployment.backend_endpoints)
    retry_after_seconds = deployment.launch_policy.queue_backpressure.retry_after_seconds
    retry_arguments = ("--retry-after-seconds", str(retry_after_seconds)) if retry_after_seconds is not None else ()
    return (
        "python3",
        get_container_path(plan, runtime_proxy_path.as_posix()),
        "--listen-port",
        str(port),
        *retry_arguments,
        *(argument for backend in backends for argument in ("--backend", backend)),
    )


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
    _validate_local_vllm_process(process)
    command = _build_vllm_command(deployment, process)
    environment, container_environment = _build_vllm_environment(deployment, source_environment)
    return _build_srun_step(
        step_id=process.process_id,
        role=RuntimeStepRole.SERVER,
        image_path=deployment.image.path,
        command=command,
        environment=environment,
        container_environment=container_environment,
        plan=plan,
        attempt_directory=attempt_directory,
        gpu_indices=tuple(process.gpu_indices),
    )


def _validate_local_vllm_process(process: ResolvedVllmProcess) -> None:
    if process.pipeline_parallel != 1 or process.node_index != 0 or process.http_port is None:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "one-node runtime received a distributed vLLM process",
        )


def _build_vllm_command(
    deployment: ResolvedVllmServerDeployment,
    process: ResolvedVllmProcess,
) -> tuple[str, ...]:
    if process.http_port is None:  # pragma: no cover - validated before command construction
        raise AssertionError("vLLM HTTP port is unavailable")
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
    return command + deployment.launch_policy.extra_args


def _build_vllm_environment(
    deployment: ResolvedVllmServerDeployment,
    source_environment: Mapping[str, str],
) -> tuple[dict[str, str], tuple[str, ...]]:
    environment = _base_environment(source_environment)
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
    return environment, tuple(container_environment)


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
    srun_command = _build_srun_prefix(plan, image_path)
    _add_srun_resources(srun_command, gpu_indices)
    _add_srun_container_options(srun_command, plan, container_environment)
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


def _build_srun_prefix(plan: ResolvedSlurmRunPlan, image_path: str) -> list[str]:
    return [
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


def _add_srun_resources(srun_command: list[str], gpu_indices: tuple[int, ...]) -> None:
    if gpu_indices:
        srun_command.extend(
            (
                f"--gpus-per-task={len(gpu_indices)}",
                f"--gpu-bind=mask_gpu:{_get_gpu_mask(gpu_indices):#x}",
            )
        )
    else:
        srun_command.append("--gres=none")


def _add_srun_container_options(
    srun_command: list[str],
    plan: ResolvedSlurmRunPlan,
    container_environment: tuple[str, ...],
) -> None:
    mounts = _render_mounts(plan)
    if mounts:
        srun_command.append(f"--container-mounts={mounts}")
    if container_environment:
        srun_command.append(f"--container-env={','.join(sorted(set(container_environment)))}")


def _get_gpu_mask(gpu_indices: tuple[int, ...]) -> int:
    return sum(1 << index for index in gpu_indices)


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
    _collect_secret_environment_names(plan.client.authored.dependencies.index_credentials, names)
    _collect_secret_environment_names(plan.invocation.authored.mcp_providers, names)
    return tuple(sorted(names))


def _collect_secret_environment_names(value: object, names: set[str]) -> None:
    if isinstance(value, SecretRef):
        names.add(value.environment)
    elif isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _collect_secret_environment_names(getattr(value, field_name), names)
    elif isinstance(value, Mapping):
        for child in value.values():
            _collect_secret_environment_names(child, names)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _collect_secret_environment_names(child, names)
