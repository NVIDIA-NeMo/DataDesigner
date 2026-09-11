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
    SlurmRunExecution,
    SlurmServiceError,
    SlurmServiceErrorCode,
    SlurmServiceOperation,
)


class _RunService:
    def __init__(self) -> None:
        self.calls: list[tuple[DataDesignerSlurmConfig, Path, bool, bool]] = []

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


def test_profile_init_creates_starter_and_emits_validation_command(tmp_path: Path) -> None:
    profile_file = tmp_path / "profile.yml"
    workspace = tmp_path / "workspace"

    result = CliRunner().invoke(
        cli_module.create_cli(),
        [
            "profile",
            "init",
            "--workspace-root",
            str(workspace),
            "--image-build-partition",
            "cpu",
            "--profile-file",
            str(profile_file),
            "--host-pattern",
            "login.example.test",
        ],
    )

    assert result.exit_code == 0, result.output
    assert json.loads(result.stdout) == {
        "profile_file": profile_file.as_posix(),
        "validation_command": f"data-designer slurm profile validate --profile-file {profile_file.as_posix()}",
    }
    assert profile_file.is_file()
    assert not workspace.exists()


def test_profile_validate_emits_selected_effective_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile_file = tmp_path / "profile.json"
    profile_file.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "default_cluster": "local",
                "clusters": {
                    "local": {
                        "schema_version": 1,
                        "gpus_per_node": 4,
                        "workspace_root": workspace.as_posix(),
                        "image_build": {
                            "partition": "cpu",
                            "cpus_per_task": 2,
                            "memory": "8G",
                            "time_limit": "04:00:00",
                        },
                    }
                },
            }
        )
    )

    result = CliRunner().invoke(
        cli_module.create_cli(),
        ["profile", "validate", "--profile-file", str(profile_file), "--cluster", "local"],
    )

    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["profile_file"] == profile_file.as_posix()
    assert payload["selected_cluster"] == "local"
    assert payload["selection_source"] == "explicit"
    assert payload["workspace_root"] == workspace.as_posix()
    assert payload["image_root"] == (workspace / "images").as_posix()
    assert payload["registry_file"] == (workspace / "images" / "registry.yaml").as_posix()
    assert payload["gpus_per_node"] == 4
    assert tuple(workspace.iterdir()) == ()


def test_image_add_rejects_credential_bearing_oci_source() -> None:
    source = f"https://user:secret@registry.example/image@sha256:{'a' * 64}"

    result = CliRunner().invoke(cli_module.create_cli(), ["image", "add", source, "--kind", "client"])

    assert result.exit_code == 2
    error = json.loads(result.stderr)["error"]
    assert error == {
        "code": "invalid_request",
        "message": "OCI image source must be a credential-free registry reference without a scheme",
        "operation": "add_image",
    }
    assert "secret" not in result.stderr


def test_cli_exposes_only_m2_run_commands() -> None:
    result = CliRunner().invoke(cli_module.create_cli(), ["--help"])

    assert result.exit_code == 0
    assert all(command in result.stdout for command in ("execute", "status", "cancel", "image", "profile"))
    assert all(command not in result.stdout for command in ("retry", "merge", "benchmark"))
