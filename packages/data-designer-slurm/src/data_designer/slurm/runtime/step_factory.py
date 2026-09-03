# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compose validated allocation-local commands into structured ``srun`` steps."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import RuntimeStep, RuntimeStepRole
from data_designer.slurm.runtime.network import validate_host_name

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


@dataclass(frozen=True, slots=True)
class SrunStepOptions:
    """Scheduler and container options for one structured runtime step."""

    image_path: str
    container_environment: tuple[str, ...] = ()
    node_hosts: tuple[str, ...] = ()
    gpu_count: int | None = None
    kill_on_bad_exit: bool = False

    def __post_init__(self) -> None:
        if type(self.image_path) is not str or not self.image_path:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime image path is invalid")
        if type(self.container_environment) is not tuple or any(
            type(name) is not str or not name or any(character in name for character in (",", "=", "\0"))
            for name in self.container_environment
        ):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.INVALID_CONTEXT,
                "runtime container environment names are invalid",
            )
        if type(self.node_hosts) is not tuple:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime node hosts are invalid")
        try:
            for host in self.node_hosts:
                validate_host_name(host)
        except ValueError as error:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, str(error)) from error
        if self.node_hosts and len(self.node_hosts) != len(set(self.node_hosts)):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime node hosts must be unique")
        if self.gpu_count is not None and (type(self.gpu_count) is not int or self.gpu_count <= 0):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime GPU count is invalid")
        if type(self.kill_on_bad_exit) is not bool:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, "runtime failure policy is invalid")


def build_srun_step(
    *,
    step_id: str,
    role: RuntimeStepRole,
    command: tuple[str, ...],
    environment: Mapping[str, str],
    plan: ResolvedSlurmRunPlan,
    attempt_directory: Path,
    options: SrunStepOptions,
) -> RuntimeStep:
    """Build one shell-free, resource-scoped ``srun`` step."""
    srun_command = _build_srun_prefix(plan, options)
    _add_srun_container_options(srun_command, plan, options.container_environment)
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


def base_environment(source_environment: Mapping[str, str]) -> dict[str, str]:
    """Return the bounded environment shared by all package-owned steps."""
    environment = {"PATH": source_environment.get("PATH") or os.defpath, "LC_ALL": "C"}
    environment.update(
        (name, source_environment[name]) for name in _SLURM_ENVIRONMENT_NAMES if name in source_environment
    )
    return environment


def place_step_on_host(step: RuntimeStep, host: str) -> RuntimeStep:
    """Constrain a package-built single-task ``srun`` step to one verified host."""
    try:
        validate_host_name(host)
    except ValueError as error:
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.INVALID_CONTEXT, str(error)) from error
    if step.command[0] != "srun":
        return step
    command = list(step.command)
    command.insert(1, f"--nodelist={host}")
    return RuntimeStep(
        step_id=step.step_id,
        role=step.role,
        command=tuple(command),
        environment=step.environment,
        stdout_path=step.stdout_path,
        stderr_path=step.stderr_path,
    )


def _build_srun_prefix(plan: ResolvedSlurmRunPlan, options: SrunStepOptions) -> list[str]:
    task_count = len(options.node_hosts) or 1
    command = [
        "srun",
        f"--nodes={task_count}",
        f"--ntasks={task_count}",
        "--ntasks-per-node=1",
        "--exact",
        "--overlap",
        "--unbuffered",
        "--export=ALL",
        f"--cpus-per-task={plan.client.authored.cpus}",
        f"--container-image={options.image_path}",
    ]
    if options.node_hosts:
        command.append(f"--nodelist={','.join(options.node_hosts)}")
    if options.kill_on_bad_exit:
        command.append("--kill-on-bad-exit=1")
    if options.gpu_count is None:
        command.append("--gres=none")
    else:
        command.extend((f"--gpus-per-task={options.gpu_count}", "--gpu-bind=none"))
    return command


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


__all__ = ["SrunStepOptions", "base_environment", "build_srun_step", "place_step_on_host"]
