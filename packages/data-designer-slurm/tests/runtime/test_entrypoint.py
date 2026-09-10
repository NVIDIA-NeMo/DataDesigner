# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import sys
import venv
import zipfile
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pyarrow.parquet as pq
import pytest
from conftest import FakeStateStore, RuntimeCase
from packaging.tags import interpreter_name, interpreter_version

import data_designer.slurm.runtime.entrypoint as entrypoint
from data_designer.slurm.client.process import ClientWorkerProcess
from data_designer.slurm.client.records import ClientEnvironmentManifest, ClientInstallerOutcome, ClientResult
from data_designer.slurm.contracts import (
    ArtifactReference,
    InstalledDistribution,
    compute_canonical_json_sha256,
    compute_serialized_json_sha256,
)
from data_designer.slurm.planning import ResolvedDependencyLock, ResolvedSlurmRunPlan
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.ports import resolve_allocation_plan
from data_designer.slurm.state import AttemptLifecycleState, CandidateOutputManifest, ReadinessState


class _InjectedFailure(Exception):
    pass


@dataclass(frozen=True)
class _PluginRuntimeCase:
    context: AllocationContext
    plan_path: Path
    image_metadata_directory: Path


def test_entrypoint_rejects_relative_paths_without_traceback(capsys: pytest.CaptureFixture[str]) -> None:
    assert (
        entrypoint.main(
            (
                "prepare",
                "--plan",
                "resolved-plan.json",
                "--attempt-dir",
                "attempt-0001",
                "--runtime-root",
                "runtime",
                "--manifest",
                "runtime-manifest.json",
                "--node-host",
                "compute-001",
            )
        )
        == 64
    )
    captured = capsys.readouterr()
    assert "runtime paths must be absolute" in captured.err
    assert "Traceback" not in captured.err


def test_container_phases_use_the_container_attempt_directory(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    container_attempt_directory = Path("/container/workspace/runs/run-single/shards/shard-00000/attempts/attempt-0001")
    _patch_runtime_context(monkeypatch, runtime_case, state)

    def verify_attempt_directory(path: Path) -> None:
        assert path == container_attempt_directory
        raise _InjectedFailure

    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_attempt_directory", verify_attempt_directory)
    prepare = entrypoint._parse_arguments(
        _phase_arguments(
            "prepare",
            runtime_case,
            runtime_case.context.attempt_directory / "runtime",
            container_attempt_directory / "runtime-manifest.json",
            attempt_directory=container_attempt_directory,
        )
    )
    with pytest.raises(_InjectedFailure):
        entrypoint._prepare(prepare, {})

    def load_candidate(*args: object, attempt_directory: Path | None = None) -> None:
        assert attempt_directory == container_attempt_directory
        raise _InjectedFailure

    monkeypatch.setattr(entrypoint, "load_complete_client_candidate", load_candidate)
    client = entrypoint._parse_arguments(
        _phase_arguments("client", runtime_case, attempt_directory=container_attempt_directory)
    )
    with pytest.raises(_InjectedFailure):
        entrypoint._client(client, {}, client_worker=ClientWorkerProcess(executor=lambda command: 0))


def test_client_phase_installs_and_runs_plugin_in_default_worker_process(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
    fake_plugin_overlay: Path,
    tmp_path: Path,
) -> None:
    case = _prepare_plugin_runtime_case(runtime_case, fake_plugin_overlay, tmp_path)
    state = FakeStateStore(case.context.attempt)
    runtime_case.context = case.context
    _patch_runtime_context(monkeypatch, runtime_case, state)
    assert "data_designer.config.column_types" in sys.modules
    _use_controlled_image_inventory(monkeypatch, case.image_metadata_directory)

    allocation = resolve_allocation_plan(case.context.plan, {"SLURM_JOB_GPUS": "0"})
    endpoint = f"generator=http://127.0.0.1:{allocation.client.ports[0].port}/v1"
    worker_arguments = (
        "preflight",
        "--plan",
        case.plan_path.as_posix(),
        "--shard-id",
        case.context.shard.shard_id,
        "--attempt-id",
        case.context.attempt.attempt_id,
        "--attempt-dir",
        case.context.attempt_directory.as_posix(),
        "--endpoint",
        endpoint,
    )
    assert ClientWorkerProcess().run(worker_arguments) == 0

    client = entrypoint._parse_arguments((*_phase_arguments("client", runtime_case), "--endpoint", endpoint))
    entrypoint._client(client, {})

    environment = ClientEnvironmentManifest.model_validate_json(
        (case.context.attempt_directory / "client-environment.json").read_text()
    )
    result = ClientResult.model_validate_json((case.context.attempt_directory / "client-result.json").read_text())
    candidate = CandidateOutputManifest.model_validate_json(
        (case.context.attempt_directory / "output-manifest.json").read_text()
    )
    dataset = pq.read_table(Path(candidate.dataset_path) / candidate.files[0].relative_path)

    assert environment.installer_outcome is ClientInstallerOutcome.INSTALLED
    assert [plugin.plugin_name for plugin in environment.plugins] == ["fake-slurm-column"]
    assert result.candidate_output_manifest is not None
    assert result.candidate_output_manifest.sha256 == candidate.compute_sha256()
    assert state.attempt.candidate_output == result.candidate_output_manifest
    assert set(dataset.column_names) == {"record_id", "custom"}
    assert dataset.column("custom").to_pylist() == ["plugin-marker"] * case.context.shard.requested_records


def test_control_phases_record_running_ready_and_failed(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    runtime_root = runtime_case.context.attempt_directory / "runtime"
    manifest_path = runtime_case.context.attempt_directory / "runtime-manifest.json"
    _patch_runtime_context(monkeypatch, runtime_case, state)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_attempt_directory", lambda path: None)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_ports", lambda *args: None)
    monkeypatch.setattr(
        entrypoint,
        "build_runtime_manifest",
        lambda *args, **kwargs: SimpleNamespace(serialize_json=lambda: "{}"),
    )

    assert entrypoint.main(_phase_arguments("prepare", runtime_case, runtime_root, manifest_path)) == 0
    assert state.attempt.state is AttemptLifecycleState.RUNNING
    assert [item.state for item in state.readiness] == [ReadinessState.PENDING, ReadinessState.STARTING]
    assert manifest_path.read_text() == "{}"

    assert entrypoint.main(_phase_arguments("ready", runtime_case)) == 0
    assert state.readiness[-1].state is ReadinessState.READY

    assert entrypoint.main(_phase_arguments("fail", runtime_case)) == 0
    assert state.attempt.state is AttemptLifecycleState.FAILED
    assert [item.state for item in state.readiness[-2:]] == [ReadinessState.FAILED, ReadinessState.STOPPED]


def test_succeed_phase_stops_runtime_and_finalizes_winner(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    runtime_root = runtime_case.context.attempt_directory / "runtime"
    manifest_path = runtime_case.context.attempt_directory / "runtime-manifest.json"
    _patch_runtime_context(monkeypatch, runtime_case, state)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_attempt_directory", lambda path: None)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_ports", lambda *args: None)
    monkeypatch.setattr(
        entrypoint,
        "build_runtime_manifest",
        lambda *args, **kwargs: SimpleNamespace(serialize_json=lambda: "{}"),
    )
    assert entrypoint.main(_phase_arguments("prepare", runtime_case, runtime_root, manifest_path)) == 0
    assert entrypoint.main(_phase_arguments("ready", runtime_case)) == 0
    state.attempt = state.attempt.model_copy(
        update={
            "candidate_output": ArtifactReference(
                path=(runtime_case.context.attempt_directory / "output-manifest.json").as_posix(),
                sha256="a" * 64,
            )
        }
    )

    assert entrypoint.main(_phase_arguments("succeed", runtime_case)) == 0

    assert state.attempt.state is AttemptLifecycleState.SUCCEEDED
    assert state.readiness[-1].state is ReadinessState.STOPPED
    assert state.winners[0].attempt_id == state.attempt.attempt_id


def test_succeed_phase_does_not_strand_success_when_winner_finalization_fails(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
) -> None:
    class FailingState(FakeStateStore):
        def finalize_winner(self, *args: object, **kwargs: object) -> None:
            raise RuntimeError("injected finalization failure")

    state = FailingState(runtime_case.context.attempt)
    runtime_root = runtime_case.context.attempt_directory / "runtime"
    manifest_path = runtime_case.context.attempt_directory / "runtime-manifest.json"
    _patch_runtime_context(monkeypatch, runtime_case, state)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_attempt_directory", lambda path: None)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_ports", lambda *args: None)
    monkeypatch.setattr(
        entrypoint,
        "build_runtime_manifest",
        lambda *args, **kwargs: SimpleNamespace(serialize_json=lambda: "{}"),
    )
    assert entrypoint.main(_phase_arguments("prepare", runtime_case, runtime_root, manifest_path)) == 0
    assert entrypoint.main(_phase_arguments("ready", runtime_case)) == 0
    state.attempt = state.attempt.model_copy(
        update={
            "candidate_output": ArtifactReference(
                path=(runtime_case.context.attempt_directory / "output-manifest.json").as_posix(),
                sha256="a" * 64,
            ),
        }
    )
    assert entrypoint.main(_phase_arguments("succeed", runtime_case)) == 70
    assert state.attempt.state is AttemptLifecycleState.RUNNING


def _patch_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
    state: FakeStateStore,
) -> None:
    monkeypatch.setattr(entrypoint, "load_allocation_context", lambda *args, **kwargs: (runtime_case.context, state))
    monkeypatch.setattr(entrypoint, "get_container_path", lambda plan, path, **kwargs: path)
    monkeypatch.setenv("SLURM_JOB_GPUS", "0")


def _prepare_plugin_runtime_case(
    runtime_case: RuntimeCase,
    fake_plugin_overlay: Path,
    tmp_path: Path,
) -> _PluginRuntimeCase:
    image_distributions = runtime_case.context.plan.client.image.inspection_facts.distributions
    wheel_path = Path(runtime_case.context.plan.authored_config.path).parent / (
        "dependencies/fake_data_designer_plugin-1.0.0-py3-none-any.whl"
    )
    _build_plugin_wheel(fake_plugin_overlay, wheel_path)
    payload = runtime_case.context.plan.model_dump(mode="json")
    _configure_identity_mount(payload, runtime_case.workspace)
    _configure_plugin_builder(payload)
    client = cast(dict[str, object], payload["client"])
    _configure_client_inspection(client, image_distributions, _write_test_installer(tmp_path))
    lock = _plugin_dependency_lock(client, image_distributions, wheel_path)
    _write_dependency_lock(client, lock)

    plan = ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))
    plan_path = Path(plan.authored_config.path).with_name("resolved-plan.json")
    plan_path.write_text(plan.serialize_json())
    Path(plan.invocation.effective_input_bindings.managed_assets_path).mkdir(parents=True, exist_ok=True)
    attempt = runtime_case.context.attempt.model_copy(
        update={
            "resolved_plan": ArtifactReference(path=plan_path.as_posix(), sha256=plan.compute_sha256()),
            "state": AttemptLifecycleState.RUNNING,
        }
    )
    context = AllocationContext(plan, plan.shards[0], attempt, runtime_case.context.attempt_directory)
    image_metadata_directory = _write_inventory_bootstrap(image_distributions, tmp_path)
    return _PluginRuntimeCase(context, plan_path, image_metadata_directory)


def _configure_identity_mount(payload: dict[str, object], workspace: Path) -> None:
    mount = {"source": workspace.as_posix(), "target": workspace.as_posix(), "read_only": False}
    payload["container_mounts"] = [mount]
    selected_profile = cast(dict[str, object], payload["selected_profile"])
    profile = cast(dict[str, object], selected_profile["profile"])
    profile["container_mounts"] = [mount]
    selected_profile["profile_sha256"] = compute_canonical_json_sha256(profile)


def _configure_plugin_builder(payload: dict[str, object]) -> None:
    resolved_builder = cast(dict[str, object], payload["builder"])
    builder = cast(dict[str, object], resolved_builder["inline"])
    cast(dict[str, object], builder["data_designer"])["columns"] = [
        {"name": "record_id", "column_type": "sampler", "sampler_type": "uuid", "params": {}},
        {"name": "custom", "column_type": "fake-slurm-column"},
    ]
    resolved_builder["content_sha256"] = compute_serialized_json_sha256(builder)


def _configure_client_inspection(
    client: dict[str, object],
    image_distributions: tuple[InstalledDistribution, ...],
    installer: Path,
) -> None:
    inspection = cast(
        dict[str, object],
        cast(dict[str, object], cast(dict[str, object], client["image"])["inspection"])["inspection"],
    )
    inspection["python_abi"] = f"{interpreter_name()}{interpreter_version()}"
    inspection["python_version"] = sys.version.split()[0]
    inspection["installer_path"] = installer.as_posix()
    inspection["distributions"] = [item.model_dump(mode="json") for item in image_distributions]
    dependencies = cast(dict[str, object], cast(dict[str, object], client["authored"])["dependencies"])
    dependencies["requirements"] = ["fake-data-designer-plugin==1.0.0"]


def _plugin_dependency_lock(
    client: dict[str, object],
    image_distributions: tuple[InstalledDistribution, ...],
    wheel_path: Path,
) -> ResolvedDependencyLock:
    inspection = cast(
        dict[str, object],
        cast(dict[str, object], cast(dict[str, object], client["image"])["inspection"])["inspection"],
    )
    return ResolvedDependencyLock.model_validate(
        {
            "schema_version": 1,
            "resolver_version": "resolver-1",
            "python_abi": inspection["python_abi"],
            "client_image_sha256": cast(dict[str, object], client["image"])["sha256"],
            "authored_requirements": ("fake-data-designer-plugin==1.0.0",),
            "authored_source": None,
            "source": None,
            "image_distributions": image_distributions,
            "overlay_packages": (
                {
                    "name": "fake-data-designer-plugin",
                    "version": "1.0.0",
                    "artifact": {
                        "path": wheel_path.as_posix(),
                        "sha256": hashlib.sha256(wheel_path.read_bytes()).hexdigest(),
                    },
                },
            ),
        }
    )


def _write_dependency_lock(client: dict[str, object], lock: ResolvedDependencyLock) -> None:
    lock_path = Path(cast(dict[str, object], client["dependency_lock"])["path"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(lock.serialize_json())
    cast(dict[str, object], client["dependency_lock"])["sha256"] = lock.compute_sha256()


def _build_plugin_wheel(source: Path, destination: Path) -> None:
    """Package the installed-layout fixture as the locked wheel under test."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    dist_info = "fake_data_designer_plugin-1.0.0.dist-info"
    members = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc"
    }
    members[f"{dist_info}/WHEEL"] = (
        b"Wheel-Version: 1.0\nGenerator: data-designer-tests\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
    )
    record_path = f"{dist_info}/RECORD"
    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    for name in (*sorted(members), record_path):
        writer.writerow((name, "", ""))
    members[record_path] = output.getvalue().encode()
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in sorted(members.items()):
            archive.writestr(name, content)


def _write_test_installer(tmp_path: Path) -> Path:
    """Create an offline pip executable without changing the test checkout."""
    environment = tmp_path / "installer-environment"
    venv.EnvBuilder(with_pip=True, symlinks=True).create(environment)
    return environment / "bin/pip"


def _write_inventory_bootstrap(
    distributions: tuple[InstalledDistribution, ...],
    tmp_path: Path,
) -> Path:
    """Present plan-recorded image metadata while imports use the editable checkout."""
    bootstrap_directory = tmp_path / "client-bootstrap"
    image_metadata_directory = bootstrap_directory / "image-metadata"
    image_metadata_directory.mkdir(parents=True)
    for distribution in distributions:
        metadata = image_metadata_directory / (
            f"{distribution.name.replace('-', '_')}-{distribution.version}.dist-info/METADATA"
        )
        metadata.parent.mkdir()
        metadata.write_text(f"Metadata-Version: 2.1\nName: {distribution.name}\nVersion: {distribution.version}\n")
    (bootstrap_directory / "sitecustomize.py").write_text(
        """from __future__ import annotations

import importlib.metadata
import os
import sys

_distributions = importlib.metadata.distributions


def _controlled_distributions(**kwargs):
    if kwargs.get("path") is not None:
        return _distributions(**kwargs)
    paths = [os.environ["DATA_DESIGNER_TEST_IMAGE_METADATA"]]
    paths.extend(path for path in sys.path if path.endswith("/client-env/site-packages"))
    return _distributions(path=paths)


importlib.metadata.distributions = _controlled_distributions
"""
    )
    return image_metadata_directory


def _use_controlled_image_inventory(monkeypatch: pytest.MonkeyPatch, image_metadata_directory: Path) -> None:
    """Limit child-process inventory discovery to the plan image and installed overlay."""
    bootstrap_directory = image_metadata_directory.parent
    python_path = os.environ.get("PYTHONPATH")
    paths = (bootstrap_directory.as_posix(),) if python_path is None else (bootstrap_directory.as_posix(), python_path)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    monkeypatch.setenv("DATA_DESIGNER_TEST_IMAGE_METADATA", image_metadata_directory.as_posix())


def _phase_arguments(
    operation: str,
    runtime_case: RuntimeCase,
    runtime_root: Path | None = None,
    manifest_path: Path | None = None,
    *,
    attempt_directory: Path | None = None,
) -> tuple[str, ...]:
    arguments = (
        operation,
        "--plan",
        (runtime_case.workspace / "runs/run-single/resolved-plan.json").as_posix(),
        "--attempt-dir",
        (attempt_directory or runtime_case.context.attempt_directory).as_posix(),
    )
    if operation == "prepare":
        assert runtime_root is not None and manifest_path is not None
        return (
            *arguments,
            "--runtime-root",
            runtime_root.as_posix(),
            "--manifest",
            manifest_path.as_posix(),
            "--node-host",
            "compute-001",
        )
    return arguments
