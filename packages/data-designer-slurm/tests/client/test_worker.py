# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import replace
from functools import partial
from pathlib import Path

import pytest
from conftest import ClientWorkerCase, FakeDataDesigner

from data_designer.config import ResumeMode
from data_designer.slurm.client.errors import ClientWorkerError
from data_designer.slurm.client.execution import ClientWorker
from data_designer.slurm.client.records import (
    ClientEnvironmentManifest,
    ClientEnvironmentOutcome,
    ClientErrorCode,
    ClientOutcome,
    ClientProgress,
    ClientProgressPhase,
    ClientResult,
)
from data_designer.slurm.contracts import compute_serialized_json_sha256
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.state import CandidateOutputManifest


def test_preflight_materializes_endpoint_and_ready_environment(client_worker_case: ClientWorkerCase) -> None:
    designers: list[FakeDataDesigner] = []

    def factory(**kwargs: object) -> FakeDataDesigner:
        designer = FakeDataDesigner(**kwargs)
        designers.append(designer)
        return designer

    manifest = ClientWorker(data_designer_factory=factory).preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert manifest.outcome is ClientEnvironmentOutcome.READY
    assert designers[0].validated_builder is not None
    model_config = designers[0].validated_builder.model_configs[0]
    assert model_config.model == client_worker_case.plan.deployments[0].served_model_name
    assert model_config.provider == "slurm-generator"
    assert designers[0].initialization["model_providers"][0].endpoint == next(
        iter(client_worker_case.endpoints.values())
    )


def test_preflight_rejects_missing_managed_assets(client_worker_case: ClientWorkerCase) -> None:
    Path(client_worker_case.plan.invocation.effective_input_bindings.managed_assets_path).rmdir()

    with pytest.raises(ClientWorkerError) as error:
        ClientWorker(data_designer_factory=FakeDataDesigner).preflight(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    assert error.value.code is ClientErrorCode.CONFIG_INVALID
    manifest = ClientEnvironmentManifest.model_validate_json(
        (client_worker_case.attempt_dir / "client-environment.json").read_text()
    )
    assert manifest.outcome is ClientEnvironmentOutcome.FAILED


@pytest.mark.parametrize(
    ("record_delta", "expected_outcome"),
    ((0, ClientOutcome.COMPLETE), (-1, ClientOutcome.PARTIAL)),
)
def test_run_persists_semantic_result_and_candidate(
    client_worker_case: ClientWorkerCase,
    record_delta: int,
    expected_outcome: ClientOutcome,
) -> None:
    actual_records = client_worker_case.plan.shards[0].requested_records + record_delta
    worker = ClientWorker(data_designer_factory=partial(FakeDataDesigner, actual_records=actual_records))
    worker.preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )
    result = worker.run(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert result.outcome is expected_outcome
    assert result.actual_records == actual_records
    persisted = ClientResult.model_validate_json((client_worker_case.attempt_dir / "client-result.json").read_text())
    candidate = CandidateOutputManifest.model_validate_json(
        (client_worker_case.attempt_dir / "output-manifest.json").read_text()
    )
    progress = ClientProgress.model_validate_json((client_worker_case.attempt_dir / "client-progress.json").read_text())
    assert persisted == result
    assert candidate.actual_records == actual_records
    assert result.candidate_output_manifest.sha256 == candidate.compute_sha256()
    assert progress.phase is ClientProgressPhase.COMPLETE


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    (
        (RuntimeError("credential=do-not-persist"), ClientErrorCode.GENERATION_FAILED),
        (KeyboardInterrupt(), ClientErrorCode.INTERRUPTED),
    ),
)
def test_run_normalizes_generation_failures(
    client_worker_case: ClientWorkerCase,
    failure: BaseException,
    expected_code: ClientErrorCode,
) -> None:
    worker = ClientWorker(data_designer_factory=partial(FakeDataDesigner, create_error=failure))
    worker.preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    with pytest.raises(ClientWorkerError) as error:
        worker.run(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    assert error.value.code is expected_code
    persisted = (client_worker_case.attempt_dir / "client-result.json").read_text()
    assert "do-not-persist" not in persisted
    assert ClientResult.model_validate_json(persisted).error_code == expected_code.value


def test_preflight_rejects_required_resume_without_workspace(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    payload["invocation"]["authored"]["resume"] = "always"
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())

    with pytest.raises(ClientWorkerError) as error:
        ClientWorker(data_designer_factory=FakeDataDesigner).preflight(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    assert error.value.code is ClientErrorCode.CONFIG_INVALID


def test_run_relocates_completed_resume_downgrade(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    payload["invocation"]["authored"]["resume"] = "always"
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    resume_path = Path(plan.shards[0].resume_workspace.path)
    resume_path.mkdir()
    (resume_path / "partial").touch()
    worker = ClientWorker(data_designer_factory=FakeDataDesigner)
    worker.preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    result = worker.run(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert result.effective_resume_mode == "never"
    assert not resume_path.exists()
    assert (client_worker_case.attempt_dir / "dataset").is_dir()


def test_if_possible_without_workspace_uses_attempt_dataset(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    payload["invocation"]["authored"]["resume"] = "if_possible"
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    worker = ClientWorker(data_designer_factory=FakeDataDesigner)
    worker.preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    result = worker.run(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert result.requested_resume_mode == "if_possible"
    assert result.effective_resume_mode == "never"
    assert result.dataset_path == (client_worker_case.attempt_dir / "dataset").as_posix()


def test_if_possible_interruption_preserves_workspace_for_retry(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    payload["invocation"]["authored"]["resume"] = "if_possible"
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    interrupted = ClientWorker(
        data_designer_factory=partial(FakeDataDesigner, create_error_after_batch=KeyboardInterrupt())
    )
    interrupted.preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    with pytest.raises(ClientWorkerError):
        interrupted.run(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    resume_path = Path(plan.shards[0].resume_workspace.path)
    assert (resume_path / "batch.parquet").is_file()
    attempt_dir = resume_path.parent / "attempts" / "attempt-0002"
    overlay_path = attempt_dir / "client-env" / "site-packages"
    overlay_path.mkdir(parents=True)
    prepared = replace(
        client_worker_case.prepared,
        attempt_id="attempt-0002",
        attempt_dir=attempt_dir,
        overlay_path=overlay_path,
    )
    retry = ClientWorker(data_designer_factory=partial(FakeDataDesigner, effective_resume=ResumeMode.ALWAYS))
    retry.preflight(
        client_worker_case.plan_path,
        prepared=prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    result = retry.run(
        client_worker_case.plan_path,
        prepared=prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert result.effective_resume_mode == "always"
    assert result.dataset_path == resume_path.as_posix()


def test_preflight_rejects_builder_with_missing_plugin(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    builder = payload["builder"]["inline"]
    builder["data_designer"]["columns"] = [{"name": "custom", "column_type": "missing-plugin"}]
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(builder)
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())

    with pytest.raises(ClientWorkerError) as error:
        ClientWorker(data_designer_factory=FakeDataDesigner).preflight(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    assert error.value.code is ClientErrorCode.CONFIG_INVALID


def test_preflight_rejects_plugin_secondary_model_alias(
    client_worker_case: ClientWorkerCase,
    fake_plugin_overlay: Path,
) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    builder = payload["builder"]["inline"]
    builder["data_designer"]["columns"] = [{"name": "custom", "column_type": "fake-slurm-column"}]
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(builder)
    lock_payload = client_worker_case.lock.model_dump(mode="json")
    wheel_path = (
        client_worker_case.plan_path.parent / "dependencies" / "fake_data_designer_plugin-1.0.0-py3-none-any.whl"
    )
    lock_payload["authored_requirements"] = ["fake-data-designer-plugin==1.0.0"]
    lock_payload["overlay_packages"] = [
        {
            "name": "fake-data-designer-plugin",
            "version": "1.0.0",
            "artifact": {"path": wheel_path.as_posix(), "sha256": "a" * 64},
        }
    ]
    lock = ResolvedDependencyLock.model_validate_json(json.dumps(lock_payload))
    Path(client_worker_case.plan.client.dependency_lock.path).write_text(lock.serialize_json())
    payload["client"]["dependency_lock"]["sha256"] = lock.compute_sha256()
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    script = """
import json
import sys
from pathlib import Path

import data_designer.slurm.client.worker as worker
from data_designer.slurm.client.environment import PreparedClientEnvironment
from data_designer.slurm.client.records import ClientErrorCode, ClientInstallerOutcome
from data_designer.slurm.contracts import ArtifactReference, InstalledDistribution

plan_path = Path(sys.argv[1])
attempt_dir = Path(sys.argv[2])
plan = json.loads(plan_path.read_text())
lock = json.loads(Path(plan["client"]["dependency_lock"]["path"]).read_text())
installed = tuple(
    sorted(
        (
            *(InstalledDistribution(**item) for item in lock["image_distributions"]),
            *(InstalledDistribution(name=item["name"], version=item["version"]) for item in lock["overlay_packages"]),
        ),
        key=lambda item: item.name,
    )
)
prepared = PreparedClientEnvironment(
    run_id=plan["run_id"],
    shard_id=plan["shards"][0]["shard_id"],
    attempt_id="attempt-0001",
    attempt_dir=attempt_dir,
    overlay_path=Path(sys.argv[3]),
    dependency_lock=ArtifactReference(**plan["client"]["dependency_lock"]),
    client_image_sha256=plan["client"]["image"]["sha256"],
    python_abi=lock["python_abi"],
    installer_outcome=ClientInstallerOutcome.REUSED,
    installed_distributions=installed,
)
worker.activate_environment(prepared)

from data_designer.slurm.client.plugins import discover_plugins

plugins = discover_plugins(installed)
from data_designer.slurm.client.execution import ClientWorker
from data_designer.slurm.client.errors import ClientWorkerError

try:
    ClientWorker().preflight(
        plan_path,
        prepared=prepared,
        endpoints={"generator": "http://127.0.0.1:17000/v1"},
        plugins=plugins,
    )
except ClientWorkerError as error:
    assert error.code is ClientErrorCode.CONFIG_INVALID
    assert error.redacted_message == "builder references unavailable model aliases"
else:
    raise AssertionError("preflight accepted a missing secondary model alias")
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            client_worker_case.plan_path.as_posix(),
            client_worker_case.attempt_dir.as_posix(),
            fake_plugin_overlay.as_posix(),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_preflight_rejects_profiler_model_alias(client_worker_case: ClientWorkerCase) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    builder = payload["builder"]["inline"]
    builder["data_designer"]["profilers"] = [{"model_alias": "missing"}]
    payload["builder"]["content_sha256"] = compute_serialized_json_sha256(builder)
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())

    with pytest.raises(ClientWorkerError) as error:
        ClientWorker(data_designer_factory=FakeDataDesigner).preflight(
            client_worker_case.plan_path,
            prepared=client_worker_case.prepared,
            endpoints=client_worker_case.endpoints,
            plugins=(),
        )

    assert error.value.code is ClientErrorCode.CONFIG_INVALID


def test_preflight_keeps_mcp_secret_out_of_records(
    client_worker_case: ClientWorkerCase,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = client_worker_case.plan.model_dump(mode="json")
    payload["invocation"]["authored"]["mcp_providers"] = [
        {
            "provider_type": "sse",
            "name": "private-mcp",
            "endpoint": "https://mcp.example.test/events",
            "api_key": {"type": "secret", "environment": "WORKER_SECRET"},
        }
    ]
    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    client_worker_case.plan_path.write_text(plan.serialize_json())
    monkeypatch.setenv("WORKER_SECRET", "private-value")
    designers: list[FakeDataDesigner] = []

    def factory(**kwargs: object) -> FakeDataDesigner:
        designer = FakeDataDesigner(**kwargs)
        designers.append(designer)
        return designer

    ClientWorker(data_designer_factory=factory).preflight(
        client_worker_case.plan_path,
        prepared=client_worker_case.prepared,
        endpoints=client_worker_case.endpoints,
        plugins=(),
    )

    assert designers[0].initialization["mcp_providers"][0].api_key == "private-value"
    assert "private-value" not in (client_worker_case.attempt_dir / "client-environment.json").read_text()
