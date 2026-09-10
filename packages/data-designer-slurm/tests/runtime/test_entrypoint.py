# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeStateStore, RuntimeCase

import data_designer.slurm.runtime.entrypoint as entrypoint
from data_designer.slurm.client.process import ClientWorkerProcess
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.state import AttemptLifecycleState, ReadinessState


class _InjectedFailure(Exception):
    pass


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


def test_client_phase_starts_plugin_worker_in_a_fresh_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
    fake_plugin_overlay: Path,
    tmp_path: Path,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    _patch_runtime_context(monkeypatch, runtime_case, state)
    assert "data_designer.config.column_types" in sys.modules
    original_run = subprocess.run

    def run_plugin_probe(command: tuple[str, ...]) -> int:
        assert tuple(command[:4]) == (
            sys.executable,
            "-m",
            "data_designer.slurm.client.worker",
            "run",
        )
        script = """
import sys
from pathlib import Path

import data_designer.slurm.client.worker as worker

assert "data_designer.config.column_types" not in sys.modules
from data_designer.slurm.client.environment import PreparedClientEnvironment
from data_designer.slurm.client.records import ClientInstallerOutcome
from data_designer.slurm.contracts import ArtifactReference, InstalledDistribution

prepared = PreparedClientEnvironment(
    run_id="run-test",
    shard_id="shard-00000",
    attempt_id="attempt-0001",
    attempt_dir=Path(sys.argv[2]),
    overlay_path=Path(sys.argv[1]),
    dependency_lock=ArtifactReference(path=(Path(sys.argv[2]) / "dependency-lock.json").as_posix(), sha256="a" * 64),
    client_image_sha256="b" * 64,
    python_abi="test",
    installer_outcome=ClientInstallerOutcome.REUSED,
    installed_distributions=(InstalledDistribution(name="fake-data-designer-plugin", version="1.0.0"),),
)
worker.activate_environment(prepared)

from data_designer.slurm.client.plugins import discover_plugins

plugins = discover_plugins(prepared.installed_distributions)
assert plugins[0].plugin_name == "fake-slurm-column"

from data_designer.config import DataDesignerConfigBuilder

builder = DataDesignerConfigBuilder.from_config(
    {"data_designer": {"columns": [{"name": "custom", "column_type": "fake-slurm-column"}], "model_configs": []}}
)
assert builder.get_column_configs()[0].column_type == "fake-slurm-column"
"""
        probe = original_run(
            (
                sys.executable,
                "-c",
                script,
                fake_plugin_overlay.as_posix(),
                (tmp_path / "plugin-attempt").as_posix(),
            ),
            check=False,
            capture_output=True,
            text=True,
        )
        assert probe.returncode == 0, probe.stderr
        return 0

    def load_candidate(*args: object, **kwargs: object) -> None:
        raise _InjectedFailure

    monkeypatch.setattr(entrypoint, "load_complete_client_candidate", load_candidate)
    client = entrypoint._parse_arguments(_phase_arguments("client", runtime_case))

    with pytest.raises(_InjectedFailure):
        entrypoint._client(
            client,
            {},
            client_worker=ClientWorkerProcess(executor=run_plugin_probe),
        )


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
        return (*arguments, "--runtime-root", runtime_root.as_posix(), "--manifest", manifest_path.as_posix())
    return arguments
