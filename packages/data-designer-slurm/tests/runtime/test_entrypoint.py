# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from conftest import FakeStateStore, RuntimeCase

import data_designer.slurm.runtime.entrypoint as entrypoint
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.state import AttemptLifecycleState, ReadinessState


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


def test_control_phases_record_running_ready_and_failed(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
) -> None:
    state = FakeStateStore(runtime_case.context.attempt)
    runtime_root = runtime_case.context.attempt_directory / "runtime"
    manifest_path = runtime_case.context.attempt_directory / "runtime-manifest.json"
    _patch_runtime_context(monkeypatch, runtime_case, state)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_attempt_directory", lambda path: None)
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_ports", lambda context: None)
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
    monkeypatch.setattr(entrypoint.SystemAllocationPreflight, "verify_ports", lambda context: None)
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


def _patch_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
    runtime_case: RuntimeCase,
    state: FakeStateStore,
) -> None:
    monkeypatch.setattr(entrypoint, "load_allocation_context", lambda *args: (runtime_case.context, state))
    monkeypatch.setattr(entrypoint, "get_container_path", lambda plan, path, **kwargs: path)


def _phase_arguments(
    operation: str,
    runtime_case: RuntimeCase,
    runtime_root: Path | None = None,
    manifest_path: Path | None = None,
) -> tuple[str, ...]:
    arguments = (
        operation,
        "--plan",
        (runtime_case.workspace / "runs/run-single/resolved-plan.json").as_posix(),
        "--attempt-dir",
        runtime_case.context.attempt_directory.as_posix(),
    )
    if operation == "prepare":
        assert runtime_root is not None and manifest_path is not None
        return (*arguments, "--runtime-root", runtime_root.as_posix(), "--manifest", manifest_path.as_posix())
    return arguments
