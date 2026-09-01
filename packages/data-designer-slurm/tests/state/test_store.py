# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier
from typing import TypeVar, cast

import pytest

from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, ContractValue, compute_canonical_json_sha256
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    DeploymentReadiness,
    EndpointPublicationState,
    ReadinessState,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    SlurmStateError,
    SlurmStateWriter,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state import filesystem as state_filesystem
from data_designer.slurm.state import store as state_store

_ValueT = TypeVar("_ValueT", bound=ContractValue)


@dataclass(frozen=True)
class _StateCase:
    workspace: Path
    authored_config: DataDesignerSlurmConfig
    plan: ResolvedSlurmRunPlan
    run: RunManifest
    shards: tuple[ShardManifest, ...]
    writer: SlurmStateWriter
    created_at: datetime


def test_run_initialization_is_idempotent_restrictive_and_reloadable(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)

    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) is case.run
    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) is case.run

    reader = SlurmStateWriter(case.workspace, case.plan.run_id)
    assert reader.load_run() == case.run
    assert reader.load_authored_config() == case.authored_config
    assert reader.load_resolved_plan() == case.plan
    assert reader.load_shards() == case.shards
    assert reader.load_attempts("shard-00000") == ()

    package_directories = tuple(path for path in (case.workspace / "runs").rglob("*") if path.is_dir())
    package_directories += (case.workspace / "runs",)
    package_files = tuple(path for path in (case.workspace / "runs").rglob("*") if path.is_file())
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o700 for path in package_directories)
    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in package_files)
    assert not tuple((case.workspace / "runs").rglob(".state.*.tmp"))


def test_run_commit_marker_is_published_last_and_interrupted_initialization_retries(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    original_publish = state_store.publish_immutable_text

    def interrupt_before_run_commit(
        directory_descriptor: int,
        name: str,
        content: str,
        display_path: Path,
        *,
        maximum_size: int,
    ) -> bool:
        if name == "run.json":
            assert (case.writer.run_root / "authored-config.json").is_file()
            assert (case.writer.run_root / "resolved-plan.json").is_file()
            assert (case.writer.run_root / "shards" / "shard-00000" / "shard.json").is_file()
            with pytest.raises(StateNotFoundError):
                SlurmStateWriter(case.workspace, case.plan.run_id).load_run()
            raise OSError("simulated interruption")
        return original_publish(
            directory_descriptor,
            name,
            content,
            display_path,
            maximum_size=maximum_size,
        )

    monkeypatch.setattr(state_store, "publish_immutable_text", interrupt_before_run_commit)
    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    monkeypatch.setattr(state_store, "publish_immutable_text", original_publish)
    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) == case.run
    assert case.writer.load_shards() == case.shards


def test_run_initialization_never_replaces_different_immutable_bytes(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    different_run = _validated_copy(case.run, created_at=case.created_at - timedelta(seconds=1))

    with pytest.raises(StateConflictError, match="different immutable state"):
        case.writer.initialize_run(case.authored_config, case.plan, different_run, case.shards)

    assert case.writer.load_run() == case.run


@pytest.mark.parametrize("record_kind", ("symlink", "fifo", "permissive"))
def test_run_reader_rejects_unsafe_record_types_and_modes(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    record_kind: str,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    run_path = case.writer.run_root / "run.json"
    if record_kind == "symlink":
        outside = tmp_path / "outside-run.json"
        outside.write_text(case.run.serialize_json())
        outside.chmod(0o600)
        run_path.unlink()
        run_path.symlink_to(outside)
    elif record_kind == "fifo":
        run_path.unlink()
        os.mkfifo(run_path)
    else:
        run_path.chmod(0o644)

    with pytest.raises(StateCorruptionError, match="cannot load"):
        case.writer.load_run()


def test_initialization_rejects_symlinked_package_storage_without_touching_target(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    outside = tmp_path / "outside"
    outside.mkdir()
    (case.workspace / "runs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    assert tuple(outside.iterdir()) == ()


def test_reader_rejects_package_directory_with_loosened_permissions(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    case.writer.run_root.chmod(0o755)

    with pytest.raises(StateCorruptionError, match="cannot load"):
        case.writer.load_run()


def test_attempt_creation_recovers_an_unpublished_next_directory_and_updates_monotonically(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    partial_root = case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id
    partial_root.mkdir(mode=0o700)

    assert case.writer.load_attempts(attempt.shard_id) == ()
    assert case.writer.create_attempt(attempt) == attempt
    assert case.writer.create_attempt(attempt) == attempt

    submitted = _validated_copy(
        attempt,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0),
        updated_at=case.created_at + timedelta(minutes=2),
    )
    assert case.writer.update_attempt(submitted) == submitted
    assert (
        SlurmStateWriter(case.workspace, case.plan.run_id).load_attempt(attempt.shard_id, attempt.attempt_id)
        == submitted
    )

    with pytest.raises(StateConflictError, match="monotonic transition"):
        case.writer.update_attempt(attempt)


def test_concurrent_attempt_updates_choose_one_scheduler_identity(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    case.writer.create_attempt(attempt)
    barrier = Barrier(2)

    def submit(array_job_id: int) -> AttemptManifest:
        candidate = _validated_copy(
            attempt,
            state=AttemptLifecycleState.SUBMITTED,
            scheduler=SchedulerIdentity(array_job_id=array_job_id, array_task_id=0),
            updated_at=case.created_at + timedelta(minutes=2),
        )
        barrier.wait()
        return SlurmStateWriter(case.workspace, case.plan.run_id).update_attempt(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(submit, job_id) for job_id in (4101, 4102))

    outcomes = tuple(future.result() for future in futures if future.exception() is None)
    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StateConflictError)
    assert case.writer.load_attempt(attempt.shard_id, attempt.attempt_id) == outcomes[0]


def test_scheduler_identity_cannot_be_reused_by_a_later_attempt(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    first = _submitted_attempt(case)
    case.writer.create_attempt(first)
    second = _attempt(case, ordinal=2)
    case.writer.create_attempt(second)
    reused_scheduler = _validated_copy(
        second,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=first.scheduler,
        updated_at=case.created_at + timedelta(minutes=4),
    )

    with pytest.raises(StateConflictError, match="monotonic transition"):
        case.writer.update_attempt(reused_scheduler)

    assert case.writer.load_attempt(second.shard_id, second.attempt_id) == second


def test_attempt_reader_revalidates_global_scheduler_ownership(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    first = _submitted_attempt(case)
    case.writer.create_attempt(first)
    second = _attempt(case, ordinal=2)
    case.writer.create_attempt(second)
    conflicting = _validated_copy(
        second,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=first.scheduler,
        updated_at=case.created_at + timedelta(minutes=4),
    )
    attempt_path = case.writer.run_root / "shards" / second.shard_id / "attempts" / second.attempt_id / "attempt.json"
    attempt_path.write_text(conflicting.serialize_json())

    with pytest.raises(StateCorruptionError, match="invalid persisted attempts"):
        SlurmStateWriter(case.workspace, case.plan.run_id).load_attempts(second.shard_id)


def test_failed_atomic_attempt_replacement_preserves_previous_manifest(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    case.writer.create_attempt(attempt)
    submitted = _validated_copy(
        attempt,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0),
        updated_at=case.created_at + timedelta(minutes=2),
    )

    def fail_replace(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("simulated replacement failure")

    monkeypatch.setattr(state_filesystem.os, "replace", fail_replace)
    with pytest.raises(SlurmStateError, match="cannot update"):
        case.writer.update_attempt(submitted)

    assert case.writer.load_attempt(attempt.shard_id, attempt.attempt_id) == attempt
    assert not tuple((case.writer.run_root / "shards").rglob(".state.*.tmp"))


def test_readiness_revisions_are_idempotent_monotonic_and_atomic(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    initial = _readiness(case, attempt)

    assert case.writer.write_readiness(initial) == initial
    assert case.writer.write_readiness(initial) == initial

    starting_deployment = _validated_copy(
        initial.deployments[0],
        state=ReadinessState.STARTING,
    )
    revision_two = _validated_copy(
        initial,
        revision=2,
        updated_at=case.created_at + timedelta(minutes=4),
        state=ReadinessState.STARTING,
        deployments=(starting_deployment,),
    )
    assert case.writer.write_readiness(revision_two) == revision_two
    assert (
        SlurmStateWriter(case.workspace, case.plan.run_id).load_readiness(attempt.shard_id, attempt.attempt_id)
        == revision_two
    )

    skipped = _validated_copy(revision_two, revision=4, updated_at=case.created_at + timedelta(minutes=5))
    with pytest.raises(StateConflictError, match="monotonic transition"):
        case.writer.write_readiness(skipped)
    assert case.writer.load_readiness(attempt.shard_id, attempt.attempt_id) == revision_two


def test_concurrent_readiness_writers_cannot_publish_the_same_next_revision(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    initial = _readiness(case, attempt)
    case.writer.write_readiness(initial)
    barrier = Barrier(2)

    def publish(seconds: int) -> AttemptReadiness:
        starting = _validated_copy(initial.deployments[0], state=ReadinessState.STARTING)
        candidate = _validated_copy(
            initial,
            revision=2,
            updated_at=case.created_at + timedelta(minutes=4, seconds=seconds),
            state=ReadinessState.STARTING,
            deployments=(starting,),
        )
        barrier.wait()
        return SlurmStateWriter(case.workspace, case.plan.run_id).write_readiness(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(publish, seconds) for seconds in (1, 2))

    outcomes = tuple(future.result() for future in futures if future.exception() is None)
    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StateConflictError)
    assert case.writer.load_readiness(attempt.shard_id, attempt.attempt_id) == outcomes[0]


def test_fresh_process_loads_only_persisted_state(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    command = (
        "import sys; "
        "from data_designer.slurm.state import SlurmStateWriter; "
        "record = SlurmStateWriter(sys.argv[1], sys.argv[2]).load_run(); "
        "print(record.serialize_json(), end='')"
    )

    completed = subprocess.run(
        (sys.executable, "-c", command, case.workspace.as_posix(), case.plan.run_id),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == case.run.serialize_json()


def _build_case(
    tmp_path: Path,
    authored_config: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
) -> _StateCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    relocated_plan = _relocate_plan(plan, workspace)
    created_at = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    run_root = workspace / "runs" / relocated_plan.run_id
    resolved_plan_reference = ArtifactReference(
        path=(run_root / "resolved-plan.json").as_posix(),
        sha256=relocated_plan.compute_sha256(),
    )
    run = RunManifest(
        schema_version=1,
        run_id=relocated_plan.run_id,
        created_at=created_at,
        authored_config=relocated_plan.authored_config,
        resolved_plan=resolved_plan_reference,
        shard_count=len(relocated_plan.shards),
    )
    shards = tuple(
        ShardManifest(
            schema_version=1,
            run_id=relocated_plan.run_id,
            shard_id=planned_shard.shard_id,
            shard_index=planned_shard.shard_index,
            record_range=planned_shard.record_range,
            input_partition=planned_shard.input_partition,
            resume_workspace=planned_shard.resume_workspace,
            created_at=created_at,
        )
        for planned_shard in relocated_plan.shards
    )
    return _StateCase(
        workspace=workspace,
        authored_config=authored_config,
        plan=relocated_plan,
        run=run,
        shards=shards,
        writer=SlurmStateWriter(workspace, relocated_plan.run_id),
        created_at=created_at,
    )


def _initialized_case(
    tmp_path: Path,
    authored_config: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
) -> _StateCase:
    case = _build_case(tmp_path, authored_config, plan)
    case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)
    return case


def _relocate_plan(plan: ResolvedSlurmRunPlan, workspace: Path) -> ResolvedSlurmRunPlan:
    previous_workspace = plan.selected_profile.profile.workspace_root
    payload = cast(
        dict[str, object],
        json.loads(plan.serialize_json().replace(previous_workspace, workspace.as_posix())),
    )
    selected_profile = cast(dict[str, object], payload["selected_profile"])
    profile = SlurmProfile.model_validate_json(json.dumps(selected_profile["profile"]))
    selected_profile["profile_sha256"] = compute_canonical_json_sha256(profile.model_dump(mode="json"))
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def _attempt(case: _StateCase, *, ordinal: int = 1) -> AttemptManifest:
    return AttemptManifest(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id="shard-00000",
        attempt_id=f"attempt-{ordinal:04d}",
        attempt_ordinal=ordinal,
        resolved_plan=case.run.resolved_plan,
        state=AttemptLifecycleState.CREATED,
        created_at=case.created_at + timedelta(minutes=ordinal),
        updated_at=case.created_at + timedelta(minutes=ordinal),
    )


def _submitted_attempt(case: _StateCase) -> AttemptManifest:
    attempt = _attempt(case)
    return _validated_copy(
        attempt,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0),
        updated_at=case.created_at + timedelta(minutes=2),
    )


def _readiness(case: _StateCase, attempt: AttemptManifest) -> AttemptReadiness:
    deployments = tuple(
        DeploymentReadiness(
            deployment_id=deployment.deployment_id,
            model_alias=deployment.authored.model_alias,
            state=ReadinessState.PENDING,
            expected_backends=deployment.topology.replica_count,
            ready_backends=0,
            endpoint_publication=EndpointPublicationState.PENDING,
        )
        for deployment in case.plan.deployments
    )
    return AttemptReadiness(
        schema_version=1,
        run_id=attempt.run_id,
        shard_id=attempt.shard_id,
        attempt_id=attempt.attempt_id,
        revision=1,
        updated_at=case.created_at + timedelta(minutes=3),
        state=ReadinessState.PENDING,
        deployments=deployments,
    )


def _validated_copy(record: _ValueT, **updates: object) -> _ValueT:
    payload = record.model_dump(mode="json")
    payload.update(updates)
    return type(record).model_validate_json(json.dumps(payload, default=_json_value))


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, ContractValue):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported test JSON value: {type(value).__name__}")
