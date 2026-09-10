# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import data_designer.slurm.cli as cli_module
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.services import (
    SlurmCollectionExecution,
    SlurmRetryExecution,
    SlurmRunExecution,
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)
from data_designer.slurm.state import CollectionState


class _RunService:
    def __init__(self) -> None:
        self.calls: list[tuple[DataDesignerSlurmConfig, Path, bool, bool]] = []
        self.retry_calls: list[tuple[str, tuple[str, ...] | None, str, bool, bool]] = []
        self.collection_calls: list[tuple[Path, Path, int]] = []

    def execute(
        self,
        config: DataDesignerSlurmConfig,
        *,
        source_root: Path,
        dry_run: bool,
        force: bool,
    ) -> SlurmRunExecution:
        self.calls.append((config, source_root, dry_run, force))
        return SlurmRunExecution(
            run_id="run-0001",
            state="dry_run",
            plan_sha256="1" * 64,
            shard_count=1,
            batch_script="#!/bin/bash\n",
        )

    def retry(
        self,
        run_or_job_id: str,
        *,
        shard_ids: tuple[str, ...] | None,
        resume: str,
        dry_run: bool,
        force: bool,
    ) -> SlurmRetryExecution:
        self.retry_calls.append((run_or_job_id, shard_ids, resume, dry_run, force))
        assert shard_ids is not None
        return SlurmRetryExecution(
            run_id="run-0001",
            state="dry_run",
            shard_ids=shard_ids,
            attempt_ids=tuple("attempt-0002" for _ in shard_ids),
            effective_resume_mode="always",
            batch_script="#!/bin/bash\n",
        )

    def collect(
        self,
        input_path: Path,
        *,
        destination: Path,
        num_partitions: int,
    ) -> SlurmCollectionExecution:
        self.collection_calls.append((input_path, destination, num_partitions))
        return SlurmCollectionExecution(
            run_id="run-0001",
            collection_id="collection-0001",
            state=CollectionState.SUBMITTED,
            job_id=43,
            output_path=destination.resolve().as_posix(),
            num_partitions=num_partitions,
        )


def test_execute_emits_deterministic_json_and_forwards_actions(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    monkeypatch,
) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text(authored_run_single.serialize_json())
    service = _RunService()
    monkeypatch.setattr(cli_module, "create_slurm_run_service", lambda **_: service)

    result = CliRunner().invoke(cli_module.create_cli(), ["execute", str(run_file), "--dry-run", "--force"])

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "batch_script": "#!/bin/bash\n",
        "job_id": None,
        "plan_sha256": "1" * 64,
        "run_id": "run-0001",
        "shard_count": 1,
        "state": "dry_run",
    }
    assert service.calls == [(authored_run_single, tmp_path, True, True)]


def test_retry_emits_deterministic_json_and_maps_task_ids(monkeypatch) -> None:
    service = _RunService()
    monkeypatch.setattr(cli_module, "create_slurm_run_service", lambda **_: service)

    result = CliRunner().invoke(
        cli_module.create_cli(),
        ["retry", "42", "--task-id", "1", "--task-id", "3", "--resume", "always", "--dry-run", "--force"],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "attempt_ids": ["attempt-0002", "attempt-0002"],
        "batch_script": "#!/bin/bash\n",
        "effective_resume_mode": "always",
        "job_id": None,
        "run_id": "run-0001",
        "shard_ids": ["shard-00001", "shard-00003"],
        "state": "dry_run",
    }
    assert service.retry_calls == [("42", ("shard-00001", "shard-00003"), "always", True, True)]


def test_merge_emits_collection_job_and_forwards_paths(tmp_path: Path, monkeypatch) -> None:
    service = _RunService()
    monkeypatch.setattr(cli_module, "create_slurm_run_service", lambda **_: service)
    input_path = tmp_path / "runs/run-0001"
    output_path = tmp_path / "collected"

    result = CliRunner().invoke(
        cli_module.create_cli(),
        [
            "merge",
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--num-partitions",
            "2",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "collection_id": "collection-0001",
        "job_id": 43,
        "num_partitions": 2,
        "output_path": output_path.as_posix(),
        "run_id": "run-0001",
        "state": "submitted",
    }
    assert service.collection_calls == [(input_path, output_path, 2)]


@pytest.mark.parametrize(
    ("code", "exit_code"),
    [
        (SlurmServiceErrorCode.INVALID_REQUEST, 2),
        (SlurmServiceErrorCode.NOT_FOUND, 3),
        (SlurmServiceErrorCode.CONFLICT, 4),
        (SlurmServiceErrorCode.UNAVAILABLE, 5),
    ],
)
def test_cli_emits_stable_service_error(monkeypatch, code: SlurmServiceErrorCode, exit_code: int) -> None:
    error = SlurmServiceError(
        code,
        SlurmServiceOperation.STATUS_RUN,
        "stable failure",
    )

    def fail(**_):
        raise error

    monkeypatch.setattr(cli_module, "create_slurm_run_service", fail)

    result = CliRunner().invoke(cli_module.create_cli(), ["status", "run-0001"])

    assert result.exit_code == exit_code
    assert json.loads(result.stderr) == {
        "error": {
            "code": code.value,
            "message": "stable failure",
            "operation": "status_run",
        }
    }


def test_execute_preserves_sanitized_config_diagnostic(tmp_path: Path) -> None:
    run_file = tmp_path / "run.json"
    run_file.write_text('{"schema_version":1}')

    result = CliRunner().invoke(cli_module.create_cli(), ["execute", str(run_file)])

    assert result.exit_code == 2
    error = json.loads(result.stderr)["error"]
    assert error["code"] == "invalid_request"
    assert "failed validation" in error["message"]


def test_execute_bounds_large_config_diagnostic(tmp_path: Path) -> None:
    run_file = tmp_path / f"{'x' * 240}.json"
    run_file.write_text(json.dumps({f"unknown_field_{index}": index for index in range(100)}))

    result = CliRunner().invoke(cli_module.create_cli(), ["execute", str(run_file)])

    assert result.exit_code == 2
    message = json.loads(result.stderr)["error"]["message"]
    assert len(message) == 512
    assert message.endswith("...")


@pytest.mark.parametrize(
    "source",
    ["nvcr.io/nvidia/pytorch:24.01-py3", "docker://ubuntu:22.04"],
)
def test_image_add_rejects_mutable_oci_source(source: str) -> None:
    result = CliRunner().invoke(cli_module.create_cli(), ["image", "add", source, "--kind", "client"])

    assert result.exit_code == 2
    assert json.loads(result.stderr) == {
        "error": {
            "code": "invalid_request",
            "message": "OCI image source must be digest-qualified as name@sha256:<digest>",
            "operation": "add_image",
        }
    }


def test_cli_exposes_m3c_run_commands() -> None:
    result = CliRunner().invoke(cli_module.create_cli(), ["--help"])

    assert result.exit_code == 0
    assert all(command in result.stdout for command in ("execute", "status", "cancel", "retry", "merge", "image"))
    assert "benchmark" not in result.stdout
