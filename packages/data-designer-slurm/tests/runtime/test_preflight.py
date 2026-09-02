# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from conftest import RuntimeCase

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.runtime import preflight as runtime_preflight
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.preflight import SystemAllocationPreflight, _verify_artifact


def test_scheduler_preflight_accepts_exact_one_node_gpu_shape(runtime_case: RuntimeCase) -> None:
    environment = {
        "SLURM_ARRAY_JOB_ID": "4101",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODEID": "0",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }

    SystemAllocationPreflight._verify_scheduler(runtime_case.context, environment)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("SLURM_ARRAY_TASK_ID", "1"),
        ("SLURM_ARRAY_JOB_ID", "4102"),
        ("SLURM_JOB_NUM_NODES", "2"),
        ("SLURM_NODEID", "1"),
        ("CUDA_VISIBLE_DEVICES", "0,1"),
    ),
)
def test_scheduler_preflight_rejects_mismatched_allocation_fact(
    runtime_case: RuntimeCase,
    name: str,
    value: str,
) -> None:
    environment = {
        "SLURM_ARRAY_JOB_ID": "4101",
        "SLURM_ARRAY_TASK_ID": "0",
        "SLURM_JOB_NUM_NODES": "1",
        "SLURM_NODEID": "0",
        "CUDA_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7",
    }
    environment[name] = value

    with pytest.raises(SlurmRuntimeError, match="does not match|GPU visibility"):
        SystemAllocationPreflight._verify_scheduler(runtime_case.context, environment)


def test_artifact_verification_checks_exact_bytes_and_rejects_symlink(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.bin"
    content = b"reviewed artifact"
    artifact.write_bytes(content)
    reference = ArtifactReference(path=artifact.as_posix(), sha256=hashlib.sha256(content).hexdigest())

    _verify_artifact(reference)
    artifact.write_bytes(b"changed")
    with pytest.raises(OSError, match="digest"):
        _verify_artifact(reference)

    target = tmp_path / "target.bin"
    target.write_bytes(content)
    artifact.unlink()
    artifact.symlink_to(target)
    with pytest.raises(OSError, match="regular file"):
        _verify_artifact(reference)


def test_artifact_verification_rejects_path_replacement_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact.bin"
    content = b"reviewed artifact"
    artifact.write_bytes(content)
    reference = ArtifactReference(path=artifact.as_posix(), sha256=hashlib.sha256(content).hexdigest())
    replacement = tmp_path / "replacement.bin"
    replacement.write_bytes(b"different artifact")
    original_fstat = runtime_preflight.os.fstat
    fstat_calls = 0
    replaced = False

    def replace_after_descriptor_validation(descriptor: int) -> os.stat_result:
        nonlocal fstat_calls, replaced
        status = original_fstat(descriptor)
        fstat_calls += 1
        if fstat_calls == 2:
            replaced = True
            os.replace(replacement, artifact)
        return status

    monkeypatch.setattr(runtime_preflight.os, "fstat", replace_after_descriptor_validation)
    with pytest.raises(OSError, match="replaced while it was verified"):
        _verify_artifact(reference)
    assert replaced


def test_port_preflight_detects_collision_before_launch(
    runtime_case: RuntimeCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _UnavailableSocket:
        closed: bool = False

        def __init__(self, *arguments: object) -> None:
            del arguments

        def setsockopt(self, *arguments: object) -> None:
            del arguments

        def bind(self, address: tuple[str, int]) -> None:
            del address
            raise OSError("injected port collision")

        def close(self) -> None:
            type(self).closed = True

    monkeypatch.setattr("data_designer.slurm.runtime.preflight.socket.socket", _UnavailableSocket)

    with pytest.raises(SlurmRuntimeError, match="ports are unavailable"):
        SystemAllocationPreflight._verify_ports(runtime_case.context)
    assert _UnavailableSocket.closed


def test_attempt_directory_must_be_restrictive(runtime_case: RuntimeCase) -> None:
    runtime_case.context.attempt_directory.chmod(0o755)

    with pytest.raises(SlurmRuntimeError, match="restrictive directory"):
        SystemAllocationPreflight._verify_attempt_directory(runtime_case.context.attempt_directory)
