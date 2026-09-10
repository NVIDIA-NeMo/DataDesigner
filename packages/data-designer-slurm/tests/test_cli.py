# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

import data_designer.slurm.cli as cli_module
import data_designer.slurm.cli_benchmark as benchmark_cli_module
from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.config import DataDesignerSlurmBenchmarkConfig, DataDesignerSlurmConfig
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


class _BenchmarkService:
    def __init__(self, manifest: BenchmarkManifest, report: BenchmarkReport) -> None:
        self.manifest = manifest
        self.report = report
        self.run_calls = []
        self.analysis_calls = []

    def run(self, config, *, source_root, force):
        self.run_calls.append((config, source_root, force))
        return self.manifest

    def analyze(self, benchmark_id, *, refresh_state, fail_if_incomplete):
        self.analysis_calls.append((benchmark_id, refresh_state, fail_if_incomplete))
        return self.report


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


def test_benchmark_cli_forwards_run_and_analysis_actions(
    tmp_path: Path,
    benchmark_config: DataDesignerSlurmBenchmarkConfig,
    benchmark_manifest: BenchmarkManifest,
    benchmark_report: BenchmarkReport,
    monkeypatch,
) -> None:
    benchmark_file = tmp_path / "benchmark.json"
    benchmark_file.write_text(benchmark_config.serialize_json())
    service = _BenchmarkService(benchmark_manifest, benchmark_report)
    monkeypatch.setattr(benchmark_cli_module, "create_slurm_benchmark_service", lambda **_: service)

    run_result = CliRunner().invoke(
        cli_module.create_cli(),
        ["benchmark", "run", str(benchmark_file), "--force"],
    )
    analyze_result = CliRunner().invoke(
        cli_module.create_cli(),
        ["benchmark", "analyze", "/workspace/benchmarks/benchmark-001", "--refresh", "--fail-if-incomplete"],
    )

    assert run_result.exit_code == 0
    assert json.loads(run_result.stdout)["benchmark_id"] == benchmark_manifest.benchmark_id
    assert analyze_result.exit_code == 0
    assert json.loads(analyze_result.stdout)["analysis_id"] == benchmark_report.analysis_id
    assert service.run_calls == [(benchmark_config, tmp_path, True)]
    assert service.analysis_calls == [("benchmark-001", True, True)]


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


def test_cli_exposes_benchmark_without_retry_or_merge() -> None:
    result = CliRunner().invoke(cli_module.create_cli(), ["--help"])

    assert result.exit_code == 0
    assert all(command in result.stdout for command in ("execute", "status", "cancel", "image", "benchmark"))
    assert all(command not in result.stdout for command in ("retry", "merge"))
