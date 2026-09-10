# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast allocation checks performed before any child process starts."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.network import validate_host_name
from data_designer.slurm.runtime.paths import get_container_path
from data_designer.slurm.runtime.ports import resolve_allocation_plan

_DIGEST_CHUNK_SIZE = 1024 * 1024
_GPU_COUNT_PATTERN = re.compile(r"^(?:gpu(?::[^:]+)?):([0-9]+)$")


@dataclass(frozen=True, slots=True)
class AllocationLayout:
    """Verified allocation host identities in planner-index order."""

    node_hosts: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            type(self.node_hosts) is not tuple
            or not self.node_hosts
            or len(self.node_hosts) != len(set(self.node_hosts))
        ):
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.PREFLIGHT_FAILED, "allocation node identities are invalid")
        try:
            for host in self.node_hosts:
                validate_host_name(host)
        except ValueError as error:
            raise SlurmRuntimeError(SlurmRuntimeErrorCode.PREFLIGHT_FAILED, str(error)) from error

    def get_host(self, node_index: int) -> str:
        """Return the host assigned to one planner node index."""
        if type(node_index) is not int or node_index < 0:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "resolved node index is outside the allocation",
            )
        try:
            return self.node_hosts[node_index]
        except (IndexError, TypeError):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "resolved node index is outside the allocation",
            ) from None


def validate_allocation_layout(plan: ResolvedSlurmRunPlan, layout: AllocationLayout) -> None:
    """Require one scheduler host for every contiguous planner node index."""
    node_indices = {
        plan.client.host_node_index,
        *(index for deployment in plan.deployments for index in deployment.node_indices),
    }
    if node_indices != set(range(len(layout.node_hosts))):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "allocation host identities do not match the resolved plan",
        )


class AllocationPreflight(Protocol):
    """Verify one allocation without starting package-managed processes."""

    def verify(self, context: AllocationContext, environment: Mapping[str, str]) -> None:
        """Raise a normalized error when allocation facts disagree with the plan."""
        ...


class SystemAllocationPreflight:
    """Production verification of scheduler, filesystem, GPU, and port facts."""

    def verify(self, context: AllocationContext, environment: Mapping[str, str]) -> None:
        """Verify every launch-critical fact before model services start."""
        try:
            self._verify_scheduler(context, environment)
            attempt_directory = Path(
                get_container_path(context.plan, context.attempt_directory.as_posix(), require_writable=True)
            )
            self.verify_attempt_directory(attempt_directory)
            self._verify_artifacts(context)
            self.verify_ports(context, environment)
        except SlurmRuntimeError:
            raise
        except (OSError, ValueError) as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation preflight could not verify launch inputs",
            ) from error

    @staticmethod
    def _verify_scheduler(context: AllocationContext, environment: Mapping[str, str]) -> None:
        node_indices = {
            context.plan.client.host_node_index,
            *(index for deployment in context.plan.deployments for index in deployment.node_indices),
        }
        node_count = max(node_indices) + 1
        if node_indices != set(range(node_count)):
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "resolved allocation node indices are not complete",
            )
        expected = {
            "SLURM_ARRAY_JOB_ID": context.attempt.scheduler.array_job_id,
            "SLURM_ARRAY_TASK_ID": context.shard.array_task_index,
            "SLURM_JOB_NUM_NODES": node_count,
            "SLURM_NODEID": context.plan.client.host_node_index,
        }
        for name, value in expected.items():
            if _parse_non_negative_integer(environment.get(name), name) != value:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                    f"scheduler environment {name!r} does not match the resolved plan",
                )
        visible_gpus = environment.get("CUDA_VISIBLE_DEVICES") or environment.get("SLURM_JOB_GPUS")
        if _parse_gpu_count(visible_gpus) < context.plan.resolved_gpus_per_node:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation GPU visibility does not match the resolved plan",
            )

    @staticmethod
    def verify_attempt_directory(attempt_directory: Path) -> None:
        """Require an attempt workspace accessible only to its owner."""
        status = attempt_directory.lstat()
        if not stat.S_ISDIR(status.st_mode) or status.st_mode & 0o077:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "attempt workspace is not a restrictive directory",
            )

    @staticmethod
    def _verify_artifacts(context: AllocationContext) -> None:
        plan = context.plan
        references = [
            ArtifactReference(
                path=Path(plan.authored_config.path).with_name("resolved-plan.json").as_posix(),
                sha256=plan.compute_sha256(),
            ),
            plan.runtime_bundle,
            plan.client.dependency_lock,
            ArtifactReference(path=plan.client.image.path, sha256=plan.client.image.sha256),
            *(
                ArtifactReference(path=deployment.image.path, sha256=deployment.image.sha256)
                for deployment in plan.deployments
            ),
        ]
        optional_references = [
            plan.builder.source,
            context.shard.input_partition,
        ]
        references.extend(reference for reference in optional_references if reference is not None)
        unique_references = {(reference.path, reference.sha256): reference for reference in references}
        for reference in unique_references.values():
            if any(
                reference.path == mount.source or reference.path.startswith(f"{mount.source}/")
                for mount in plan.container_mounts
            ):
                reference = reference.model_copy(update={"path": get_container_path(plan, reference.path)})
            _verify_artifact(reference)

    @staticmethod
    def verify_ports(context: AllocationContext, environment: Mapping[str, str]) -> None:
        """Verify ports owned by the local client host before nested steps start."""
        plan = resolve_allocation_plan(context.plan, environment)
        local_node_index = plan.client.host_node_index
        ports = tuple(port.port for port in plan.client.ports if port.node_index == local_node_index) + tuple(
            port.port
            for deployment in plan.deployments
            for port in deployment.ports
            if port.node_index == local_node_index
        )
        reservations: list[socket.socket] = []
        try:
            for port in ports:
                reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                reservations.append(reservation)
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                reservation.bind(("0.0.0.0", port))
        except OSError as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "one or more local allocation ports are unavailable",
            ) from error
        finally:
            for reservation in reservations:
                reservation.close()


def _verify_artifact(reference: ArtifactReference) -> None:
    path = Path(reference.path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise OSError(f"artifact {path} is not a regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise OSError(f"artifact {path} changed while it was opened")
        digest = hashlib.sha256()
        while chunk := os.read(descriptor, _DIGEST_CHUNK_SIZE):
            digest.update(chunk)
        after = os.fstat(descriptor)
        if _file_identity(opened) != _file_identity(after):
            raise OSError(f"artifact {path} changed while it was read")
        if digest.hexdigest() != reference.sha256:
            raise OSError(f"artifact {path} digest does not match the plan")
        current = path.lstat()
        if not stat.S_ISREG(current.st_mode) or _file_identity(after) != _file_identity(current):
            raise OSError(f"artifact {path} was replaced while it was verified")
    finally:
        os.close(descriptor)


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _parse_non_negative_integer(value: str | None, name: str) -> int:
    if value is None or not value.isascii() or not value.isdigit():
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            f"scheduler environment {name!r} is unavailable or invalid",
        )
    return int(value)


def _parse_gpu_count(value: str | None) -> int:
    if value is None or not value.strip():
        return 0
    normalized = value.strip()
    match = _GPU_COUNT_PATTERN.fullmatch(normalized)
    if match is not None:
        return int(match.group(1))
    values = tuple(item.strip() for item in normalized.split(","))
    if not all(values) or len(values) != len(set(values)):
        raise SlurmRuntimeError(SlurmRuntimeErrorCode.PREFLIGHT_FAILED, "GPU visibility is invalid")
    return len(values)


__all__ = [
    "AllocationLayout",
    "AllocationPreflight",
    "SystemAllocationPreflight",
    "validate_allocation_layout",
]
