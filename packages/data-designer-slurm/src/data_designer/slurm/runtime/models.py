# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Transient typed values for allocation-local process execution."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import Identifier
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.state import AttemptManifest

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


class RuntimeStepRole(str, Enum):
    """Lifecycle role of one allocation-local Slurm step."""

    CLIENT_PREFLIGHT = "client_preflight"
    SERVER_PREFLIGHT = "server_preflight"
    SERVER = "server"
    ENDPOINT = "endpoint"
    CLIENT = "client"


@dataclass(frozen=True, slots=True)
class RuntimeStep:
    """One shell-free process specification owned by the common step runner."""

    step_id: Identifier
    role: RuntimeStepRole
    command: tuple[str, ...]
    environment: Mapping[str, str]
    stdout_path: Path
    stderr_path: Path

    def __post_init__(self) -> None:
        _validate_step_identity(self.step_id)
        if not isinstance(self.role, RuntimeStepRole):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime step role is invalid")
        _validate_step_command(self.command)
        normalized_environment = _validate_step_environment(self.environment)
        _validate_step_log_paths(self.stdout_path, self.stderr_path)
        object.__setattr__(self, "environment", MappingProxyType(normalized_environment))


def _validate_step_identity(step_id: Identifier) -> None:
    try:
        normalized_step_id = _IDENTIFIER_ADAPTER.validate_python(step_id, strict=True)
    except ValidationError as error:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime step identity is invalid") from error
    if normalized_step_id != step_id:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime step identity is not canonical")


def _validate_step_command(command: tuple[str, ...]) -> None:
    if not command or any(type(argument) is not str or not argument or "\0" in argument for argument in command):
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime step command is invalid")


def _validate_step_environment(environment: Mapping[str, str]) -> dict[str, str]:
    if not isinstance(environment, Mapping):
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime environment is invalid")
    normalized_environment: dict[str, str] = {}
    for name, value in environment.items():
        if type(name) is not str or not name or "=" in name or "\0" in name:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime environment name is invalid")
        if type(value) is not str or "\0" in value:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime environment value is invalid")
        normalized_environment[name] = value
    return normalized_environment


def _validate_step_log_paths(stdout_path: Path, stderr_path: Path) -> None:
    for path in (stdout_path, stderr_path):
        if not path.is_absolute():
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime log paths must be absolute")
    if stdout_path.parent != stderr_path.parent or stdout_path == stderr_path:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.INVALID_CONTEXT,
            "runtime log paths must be distinct children of one directory",
        )


@dataclass(frozen=True, slots=True)
class RuntimeEndpoint:
    """Client-visible logical endpoint published inside one allocation."""

    model_alias: str
    served_model_name: str
    host: str
    port: int

    def __post_init__(self) -> None:
        if type(self.model_alias) is not str or not self.model_alias:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "endpoint model alias is invalid")
        if type(self.served_model_name) is not str or not self.served_model_name:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "endpoint model name is invalid")
        if self.host != "127.0.0.1":
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "logical endpoints must use loopback")
        if type(self.port) is not int or not 1 <= self.port <= 65535:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "endpoint port is invalid")


@dataclass(frozen=True, slots=True)
class AllocationContext:
    """Verified plan, shard, attempt, and package-owned attempt directory."""

    plan: ResolvedSlurmRunPlan
    shard: PlannedShard
    attempt: AttemptManifest
    attempt_directory: Path

    def __post_init__(self) -> None:
        if not isinstance(self.plan, ResolvedSlurmRunPlan):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation plan is invalid")
        if not isinstance(self.shard, PlannedShard) or self.shard not in self.plan.shards:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation shard is not in the plan")
        if not isinstance(self.attempt, AttemptManifest):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "allocation attempt is invalid")
        expected_directory = (
            Path(self.plan.authored_config.path).parent
            / "shards"
            / self.shard.shard_id
            / "attempts"
            / self.attempt.attempt_id
        )
        if self.attempt_directory != expected_directory:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.INVALID_CONTEXT,
                "allocation attempt directory does not match persisted identity",
            )
        if (
            self.attempt.run_id != self.plan.run_id
            or self.attempt.shard_id != self.shard.shard_id
            or self.attempt.scheduler is None
            or self.attempt.scheduler.array_task_id != self.shard.array_task_index
        ):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.INVALID_CONTEXT,
                "allocation attempt does not match its planned scheduler task",
            )
