# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-fast allocation checks performed before any child process starts."""

from __future__ import annotations

import hashlib
import os
import re
import socket
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.network import validate_host_name
from data_designer.slurm.runtime.paths import get_container_path

_DIGEST_CHUNK_SIZE = 1024 * 1024
_GPU_COUNT_PATTERN = re.compile(r"^(?:gpu(?::[^:]+)?):([0-9]+)$")
_NODE_LIST_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.\-,\[\]]{0,4095}$")


@dataclass(frozen=True, slots=True)
class AllocationLayout:
    """Verified allocation node identities in planner index order."""

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
        """Return the verified host assigned to one planner node index."""
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


class AllocationHostResolver(Protocol):
    """Resolve the scheduler node expression to ordered host identities."""

    def resolve(self, node_list: str, environment: Mapping[str, str]) -> tuple[str, ...]:
        """Return one host name per allocation node in scheduler order."""
        ...


class ScontrolAllocationHostResolver:
    """Resolve Slurm host expressions through the scheduler-owned CLI."""

    def resolve(self, node_list: str, environment: Mapping[str, str]) -> tuple[str, ...]:
        """Expand a validated allocation node expression without a shell."""
        if type(node_list) is not str or _NODE_LIST_PATTERN.fullmatch(node_list) is None:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "scheduler allocation node list is unavailable or invalid",
            )
        command_environment = {
            "PATH": environment.get("PATH") or os.defpath,
            "LC_ALL": "C",
        }
        try:
            completed = subprocess.run(
                ("scontrol", "show", "hostnames", node_list),
                capture_output=True,
                text=True,
                check=False,
                timeout=30,
                env=command_environment,
            )
        except (OSError, subprocess.SubprocessError) as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation host identities could not be resolved",
            ) from error
        if completed.returncode != 0:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation host identities could not be resolved",
            )
        return tuple(line.strip() for line in completed.stdout.splitlines() if line.strip())


class AllocationPreflight(Protocol):
    """Verify one allocation and resolve its node identities before launch."""

    def verify(self, context: AllocationContext, environment: Mapping[str, str]) -> AllocationLayout:
        """Return verified allocation layout or raise a normalized error."""
        ...


class SystemAllocationPreflight:
    """Production verification of scheduler, filesystem, GPU, and port facts."""

    def __init__(self, host_resolver: AllocationHostResolver | None = None) -> None:
        self._host_resolver = host_resolver or ScontrolAllocationHostResolver()

    def verify(self, context: AllocationContext, environment: Mapping[str, str]) -> AllocationLayout:
        """Verify every launch-critical fact before model services start."""
        try:
            node_count = self._verify_scheduler(context, environment)
            layout = self._resolve_layout(node_count, environment)
            self._verify_attempt_directory(context.attempt_directory)
            get_container_path(context.plan, context.attempt_directory.as_posix(), require_writable=True)
            self._verify_artifacts(context)
            verify_local_ports(context)
        except SlurmRuntimeError:
            raise
        except (OSError, ValueError) as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation preflight could not verify launch inputs",
            ) from error
        return layout

    @staticmethod
    def _verify_scheduler(context: AllocationContext, environment: Mapping[str, str]) -> int:
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
            "SLURM_NODEID": 0,
        }
        for name, value in expected.items():
            if _parse_non_negative_integer(environment.get(name), name) != value:
                raise SlurmRuntimeError(
                    SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                    f"scheduler environment {name!r} does not match the resolved plan",
                )
        visible_gpus = environment.get("CUDA_VISIBLE_DEVICES") or environment.get("SLURM_JOB_GPUS")
        if _parse_gpu_count(visible_gpus) != context.plan.resolved_gpus_per_node:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "allocation GPU visibility does not match the resolved plan",
            )
        return node_count

    def _resolve_layout(self, node_count: int, environment: Mapping[str, str]) -> AllocationLayout:
        node_list = environment.get("SLURM_JOB_NODELIST")
        if node_list is None or _NODE_LIST_PATTERN.fullmatch(node_list) is None:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "scheduler allocation node list is unavailable or invalid",
            )
        layout = AllocationLayout(self._host_resolver.resolve(node_list, environment))
        if len(layout.node_hosts) != node_count:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "scheduler allocation host count does not match the resolved plan",
            )
        return layout

    @staticmethod
    def _verify_attempt_directory(attempt_directory: Path) -> None:
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
            verify_artifact(reference)


def verify_local_ports(context: AllocationContext) -> None:
    """Verify ports owned by the local controller node before nested steps start."""
    local_node_index = context.plan.client.host_node_index
    ports = tuple(port.port for port in context.plan.client.ports if port.node_index == local_node_index) + tuple(
        port.port
        for deployment in context.plan.deployments
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
            "one or more resolved allocation ports are unavailable",
        ) from error
    finally:
        for reservation in reservations:
            reservation.close()


def verify_artifact(reference: ArtifactReference) -> None:
    """Verify one regular artifact without following or racing a replacement."""
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
    "AllocationHostResolver",
    "AllocationLayout",
    "AllocationPreflight",
    "ScontrolAllocationHostResolver",
    "SystemAllocationPreflight",
    "verify_artifact",
    "verify_local_ports",
]
