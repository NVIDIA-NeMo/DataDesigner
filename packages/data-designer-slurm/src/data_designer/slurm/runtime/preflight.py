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
from pathlib import Path
from typing import Protocol

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.paths import get_container_path

_DIGEST_CHUNK_SIZE = 1024 * 1024
_GPU_COUNT_PATTERN = re.compile(r"^(?:gpu(?::[^:]+)?):([0-9]+)$")


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
            self._verify_attempt_directory(context.attempt_directory)
            get_container_path(context.plan, context.attempt_directory.as_posix(), require_writable=True)
            self._verify_artifacts(context)
            self._verify_ports(context)
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
        if node_indices != {0}:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "one-node runtime received a multi-node execution plan",
            )
        expected = {
            "SLURM_ARRAY_JOB_ID": context.attempt.scheduler.array_job_id,
            "SLURM_ARRAY_TASK_ID": context.shard.array_task_index,
            "SLURM_JOB_NUM_NODES": 1,
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
            _verify_artifact(reference)

    @staticmethod
    def _verify_ports(context: AllocationContext) -> None:
        ports = tuple(port.port for port in context.plan.client.ports) + tuple(
            port.port for deployment in context.plan.deployments for port in deployment.ports
        )
        reservations: list[socket.socket] = []
        try:
            for port in ports:
                reservation = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                reservation.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 0)
                reservation.bind(("127.0.0.1", port))
                reservations.append(reservation)
        except OSError as error:
            raise SlurmRuntimeError(
                SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
                "one or more resolved allocation ports are unavailable",
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


__all__ = ["AllocationPreflight", "SystemAllocationPreflight"]
