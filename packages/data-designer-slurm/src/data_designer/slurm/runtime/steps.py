# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build coordinated allocation-local processes as structured ``srun`` steps."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Protocol

from pydantic import BaseModel

from data_designer.slurm.config.environment import SecretRef
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.endpoint_steps import build_endpoint_steps as build_endpoint_steps
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeEndpoint, RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.server_steps import build_vllm_steps as build_vllm_steps
from data_designer.slurm.runtime.step_factory import (
    SrunStepOptions,
    base_environment,
    build_srun_step,
)
from data_designer.slurm.runtime.step_factory import (
    place_step_on_host as place_step_on_host,
)
from data_designer.slurm.state import AttemptManifest


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
        return build_srun_step(
            step_id=step_id,
            role=role,
            command=command,
            environment=environment,
            plan=plan,
            attempt_directory=attempt_directory,
            options=SrunStepOptions(
                image_path=plan.client.image.path,
                container_environment=secret_names,
            ),
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
    environment = base_environment(source_environment)
    for name in secret_names:
        try:
            environment[name] = source_environment[name]
        except KeyError:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                f"required client secret environment {name!r} is unavailable",
            ) from None
    return secret_names, environment


def plan_path(plan: ResolvedSlurmRunPlan) -> str:
    """Return the canonical persisted resolved-plan path."""
    return str(Path(plan.authored_config.path).with_name("resolved-plan.json"))


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
