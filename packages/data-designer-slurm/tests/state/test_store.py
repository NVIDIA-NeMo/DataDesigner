# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Barrier, Event
from typing import TypeVar, cast

import pytest

from data_designer.slurm import filesystem as slurm_filesystem
from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, ContractValue, compute_canonical_json_sha256, pretty_json
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptId,
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    AttemptTerminalClassification,
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    DeploymentReadiness,
    EndpointPublicationState,
    ReadinessState,
    RunManifest,
    SchedulerIdentity,
    ShardId,
    ShardManifest,
    ShardWinner,
    SlurmStateError,
    SlurmStateWriter,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state import filesystem as state_filesystem
from data_designer.slurm.state import storage as state_storage

_ValueT = TypeVar("_ValueT", bound=ContractValue)


@dataclass(frozen=True, slots=True)
class _StateCase:
    workspace: Path
    authored_config: DataDesignerSlurmConfig
    plan: ResolvedSlurmRunPlan
    run: RunManifest
    shards: tuple[ShardManifest, ...]
    writer: SlurmStateWriter
    created_at: datetime


@dataclass(frozen=True, slots=True)
class _FinalizationCase:
    attempt: AttemptManifest
    client_result: ClientResult
    candidate: CandidateOutputManifest
    published_at: datetime
    output_path: Path


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


def test_concurrent_identical_run_initialization_converges(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    barrier = Barrier(2)

    def initialize() -> RunManifest:
        barrier.wait()
        writer = SlurmStateWriter(case.workspace, case.plan.run_id)
        return writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: initialize(), range(2)))

    assert results == (case.run, case.run)
    assert case.writer.load_run() == case.run


def test_context_loading_reads_each_immutable_record_once(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    original_read_record = state_storage.StateStorage.read_record
    record_names: list[str] = []

    def track_read(
        self: state_storage.StateStorage,
        directory_descriptor: int,
        name: str,
        display_path: Path,
        record_type: type[_ValueT],
    ) -> _ValueT:
        record_names.append(name)
        return original_read_record(self, directory_descriptor, name, display_path, record_type)

    monkeypatch.setattr(state_storage.StateStorage, "read_record", track_read)

    assert case.writer.load_attempts("shard-00000") == ()
    assert record_names.count("run.json") == 1
    assert record_names.count("authored-config.json") == 1
    assert record_names.count("resolved-plan.json") == 1
    assert record_names.count("shard.json") == 1

    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    readiness = _readiness(case, attempt)
    case.writer.write_readiness(readiness)
    record_names.clear()

    assert case.writer.load_readiness(attempt.shard_id, attempt.attempt_id) == readiness
    assert record_names.count("run.json") == 1
    assert record_names.count("authored-config.json") == 1
    assert record_names.count("resolved-plan.json") == 1
    assert record_names.count("shard.json") == 1


def test_shard_mutations_do_not_scan_unrelated_attempt_directories(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run, multi_node_plan)
    unrelated_attempts = case.writer.run_root / "shards" / "shard-00001" / "attempts"
    (unrelated_attempts / "unexpected").mkdir(mode=0o700)
    attempt = _submitted_attempt(case)

    assert case.writer.create_attempt(attempt) == attempt
    readiness = _readiness(case, attempt)
    assert case.writer.write_readiness(readiness) == readiness
    assert case.writer.load_attempts(attempt.shard_id) == (attempt,)
    with pytest.raises(StateCorruptionError, match="invalid attempt directory"):
        case.writer.load_attempts("shard-00001")


def test_shard_mutation_rejects_run_count_drift_without_scanning_other_shards(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    run_path = case.writer.run_root / "run.json"
    run_path.write_text(case.run.model_copy(update={"shard_count": 2}).serialize_json())

    with pytest.raises(StateCorruptionError, match="shard .* is invalid"):
        case.writer.create_attempt(_submitted_attempt(case))


def test_incomplete_attempt_recognizes_created_state_temporary_file(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    attempt_root = case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id
    attempt_root.mkdir(mode=0o700)
    with state_filesystem.open_verified_directory(attempt_root, require_private=True) as directory_descriptor:
        temporary_descriptor, temporary_name = state_filesystem.create_state_temporary_file(directory_descriptor)
        os.close(temporary_descriptor)

    assert state_filesystem.is_state_temporary_name(temporary_name)
    assert case.writer.load_attempts(attempt.shard_id) == ()


def test_reader_rejects_unsupported_persisted_schema_version(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    run_path = case.writer.run_root / "run.json"
    payload = json.loads(run_path.read_text())
    payload["schema_version"] = 2
    run_path.write_text(pretty_json(payload))

    with pytest.raises(StateCorruptionError, match="unsupported schema_version"):
        case.writer.load_run()


def test_reader_rejects_noncanonical_record_that_relies_on_new_defaults(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    authored_config_path = case.writer.run_root / "authored-config.json"
    payload = json.loads(authored_config_path.read_text())
    del payload["array_tasks"]
    authored_config_path.write_text(pretty_json(payload))

    with pytest.raises(StateCorruptionError, match="schema migrations are not supported"):
        case.writer.load_authored_config()


def test_run_commit_marker_is_published_last_and_interrupted_initialization_retries(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    original_publish = state_storage.publish_immutable_text

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

    monkeypatch.setattr(state_storage, "publish_immutable_text", interrupt_before_run_commit)
    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    monkeypatch.setattr(state_storage, "publish_immutable_text", original_publish)
    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) == case.run
    assert case.writer.load_shards() == case.shards


def test_interrupted_immutable_publication_recovers_the_committed_hard_link(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    original_unlink = state_filesystem.os.unlink

    def interrupt_temporary_unlink(path: str, *, dir_fd: int | None = None) -> None:
        del path, dir_fd
        raise KeyboardInterrupt("injected interruption after immutable publication")

    monkeypatch.setattr(state_filesystem.os, "unlink", interrupt_temporary_unlink)
    with pytest.raises(KeyboardInterrupt, match="after immutable publication"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    committed_path = case.writer.run_root / "authored-config.json"
    temporary_paths = tuple(case.writer.run_root.glob(".state.*.tmp"))
    assert committed_path.stat().st_nlink == 2
    assert len(temporary_paths) == 1
    assert temporary_paths[0].stat().st_ino == committed_path.stat().st_ino

    monkeypatch.setattr(state_filesystem.os, "unlink", original_unlink)
    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) == case.run
    assert committed_path.stat().st_nlink == 1
    assert not tuple(case.writer.run_root.glob(".state.*.tmp"))


def test_concurrent_recovery_does_not_make_the_immutable_publisher_fail(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    original_unlink = state_filesystem.os.unlink
    recovered = False

    def recover_before_publisher_unlink(path: str, *, dir_fd: int | None = None) -> None:
        nonlocal recovered
        assert dir_fd is not None
        monkeypatch.setattr(state_filesystem.os, "unlink", original_unlink)
        assert (
            state_filesystem.read_regular_text(
                dir_fd,
                "authored-config.json",
                case.writer.run_root / "authored-config.json",
                maximum_size=16 * 1024 * 1024,
            )
            == authored_run_single.serialize_json()
        )
        recovered = True
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(state_filesystem.os, "unlink", recover_before_publisher_unlink)
    assert case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards) == case.run
    assert recovered
    assert not tuple(case.writer.run_root.glob(".state.*.tmp"))


def test_interrupted_publication_recovery_accepts_a_concurrent_atomic_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "records"
    directory.mkdir(mode=0o700)
    record_path = directory / "record.json"
    record_path.write_text("old")
    record_path.chmod(0o600)
    temporary_path = directory / ".state.0123456789abcdef.tmp"
    os.link(record_path, temporary_path)
    replacement_path = directory / "replacement.json"
    replacement_path.write_text("new")
    replacement_path.chmod(0o600)
    original_unlink = state_filesystem.os.unlink

    def replace_after_recovery_unlink(path: str, *, dir_fd: int | None = None) -> None:
        assert dir_fd is not None
        original_unlink(path, dir_fd=dir_fd)
        os.replace(
            replacement_path.name,
            record_path.name,
            src_dir_fd=dir_fd,
            dst_dir_fd=dir_fd,
        )

    with state_filesystem.open_verified_directory(directory, require_private=True) as directory_descriptor:
        monkeypatch.setattr(state_filesystem.os, "unlink", replace_after_recovery_unlink)
        assert (
            state_filesystem.read_regular_text(
                directory_descriptor,
                record_path.name,
                record_path,
                maximum_size=16,
            )
            == "new"
        )


def test_record_reader_retries_an_atomic_replacement_between_stat_and_open(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "records"
    directory.mkdir(mode=0o700)
    record_path = directory / "record.json"
    record_path.write_text("old")
    record_path.chmod(0o600)
    replacement_path = directory / "replacement.json"
    replacement_path.write_text("new")
    replacement_path.chmod(0o600)
    original_open = state_filesystem.os.open
    replaced = False

    with state_filesystem.open_verified_directory(directory, require_private=True) as directory_descriptor:

        def replace_before_open(
            path: str | bytes,
            flags: int,
            mode: int = 0o777,
            *,
            dir_fd: int | None = None,
        ) -> int:
            nonlocal replaced
            if path == record_path.name and not replaced:
                replaced = True
                os.replace(
                    replacement_path.name,
                    record_path.name,
                    src_dir_fd=directory_descriptor,
                    dst_dir_fd=directory_descriptor,
                )
            return original_open(path, flags, mode, dir_fd=dir_fd)

        monkeypatch.setattr(state_filesystem.os, "open", replace_before_open)
        assert (
            state_filesystem.read_regular_text(
                directory_descriptor,
                record_path.name,
                record_path,
                maximum_size=16,
            )
            == "new"
        )
    assert replaced


def test_record_reader_preserves_on_disk_newlines(tmp_path: Path) -> None:
    directory = tmp_path / "records"
    directory.mkdir(mode=0o700)
    record_path = directory / "record.json"
    record_path.write_bytes(b"first\r\nsecond\r\n")
    record_path.chmod(0o600)

    with state_filesystem.open_verified_directory(directory, require_private=True) as directory_descriptor:
        assert (
            state_filesystem.read_regular_text(
                directory_descriptor,
                record_path.name,
                record_path,
                maximum_size=64,
            )
            == "first\r\nsecond\r\n"
        )


def test_immutable_publication_retry_resyncs_existing_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directory = tmp_path / "records"
    directory.mkdir(mode=0o700)
    record_path = directory / "record.json"
    original_fsync = state_filesystem.os.fsync
    directory_syncs = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 1:
                raise OSError("simulated post-publish fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(state_filesystem.os, "fsync", fail_first_directory_fsync)
    with state_filesystem.open_verified_directory(directory, require_private=True) as directory_descriptor:
        with pytest.raises(OSError, match="post-publish"):
            state_filesystem.publish_immutable_text(
                directory_descriptor,
                record_path.name,
                "persisted",
                record_path,
                maximum_size=64,
            )
        syncs_before_retry = directory_syncs
        assert not state_filesystem.publish_immutable_text(
            directory_descriptor,
            record_path.name,
            "persisted",
            record_path,
            maximum_size=64,
        )

    assert directory_syncs > syncs_before_retry


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


def test_run_reader_rejects_manifest_bound_to_a_different_run(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    mismatched = _validated_copy(case.run, run_id="other-run")
    (case.writer.run_root / "run.json").write_text(mismatched.serialize_json())

    with pytest.raises(StateCorruptionError, match="persisted location"):
        case.writer.load_run()


@pytest.mark.parametrize("record_kind", ("symlink", "hardlink", "fifo", "permissive"))
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
    elif record_kind == "hardlink":
        outside = tmp_path / "outside-run.json"
        outside.write_text(case.run.serialize_json())
        outside.chmod(0o600)
        run_path.unlink()
        os.link(outside, run_path)
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


@pytest.mark.parametrize("lock_kind", ("symlink", "fifo"))
def test_initialization_rejects_unsafe_run_lock_without_changing_target(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    lock_kind: str,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    lock_root = case.workspace / "runs" / ".locks"
    lock_root.mkdir(parents=True, mode=0o700)
    lock_path = lock_root / f"run-{case.plan.run_id}.lock"
    if lock_kind == "symlink":
        outside = tmp_path / "outside.lock"
        outside.write_text("outside")
        outside.chmod(0o640)
        lock_path.symlink_to(outside)
    else:
        os.mkfifo(lock_path)

    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    if lock_kind == "symlink":
        assert stat.S_IMODE(outside.stat().st_mode) == 0o640


def test_temporary_file_permission_failure_leaves_no_partial_record(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    original_fchmod = state_filesystem.os.fchmod
    regular_file_calls = 0

    def fail_second_regular_file(descriptor: int, mode: int) -> None:
        nonlocal regular_file_calls
        if stat.S_ISREG(os.fstat(descriptor).st_mode):
            regular_file_calls += 1
            if regular_file_calls == 2:
                raise OSError("simulated permission failure")
        original_fchmod(descriptor, mode)

    monkeypatch.setattr(state_filesystem.os, "fchmod", fail_second_regular_file)
    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    assert not tuple((case.workspace / "runs").rglob(".state.*.tmp"))
    assert not (case.writer.run_root / "authored-config.json").exists()


def test_run_lock_rejects_path_replacement_during_acquisition(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    lock_path = case.workspace / "runs" / ".locks" / f"run-{case.plan.run_id}.lock"
    original_flock = slurm_filesystem.fcntl.flock
    replaced = False

    def replace_acquired_lock(descriptor: int, operation: int) -> None:
        nonlocal replaced
        original_flock(descriptor, operation)
        if operation == slurm_filesystem.fcntl.LOCK_EX and not replaced:
            replaced = True
            lock_path.unlink()
            lock_path.write_text("")
            lock_path.chmod(0o600)

    monkeypatch.setattr(slurm_filesystem.fcntl, "flock", replace_acquired_lock)
    with pytest.raises(SlurmStateError, match="cannot initialize"):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    assert not (case.writer.run_root / "run.json").exists()


def test_run_lock_closes_descriptor_when_acquisition_is_interrupted(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _build_case(tmp_path, authored_run_single, single_node_plan)
    lock_descriptor: int | None = None

    def interrupt_flock(descriptor: int, operation: int) -> None:
        nonlocal lock_descriptor
        del operation
        lock_descriptor = descriptor
        raise KeyboardInterrupt

    monkeypatch.setattr(slurm_filesystem.fcntl, "flock", interrupt_flock)
    with pytest.raises(KeyboardInterrupt):
        case.writer.initialize_run(case.authored_config, case.plan, case.run, case.shards)

    assert lock_descriptor is not None
    with pytest.raises(OSError):
        os.fstat(lock_descriptor)


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


def test_attempt_reader_retries_when_publication_wins_missing_file_race(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    attempt_root = case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id
    attempt_root.mkdir(mode=0o700)
    original_read = state_storage.read_regular_text
    published = False

    def publish_before_reporting_missing(
        directory_descriptor: int,
        name: str,
        display_path: Path,
        *,
        maximum_size: int,
    ) -> str:
        nonlocal published
        if name == "attempt.json" and not published:
            published = True
            display_path.write_text(attempt.serialize_json())
            display_path.chmod(0o600)
            raise FileNotFoundError(name)
        return original_read(directory_descriptor, name, display_path, maximum_size=maximum_size)

    monkeypatch.setattr(state_storage, "read_regular_text", publish_before_reporting_missing)

    assert case.writer.load_attempts(attempt.shard_id) == (attempt,)
    assert published


def test_attempt_recovery_rejects_unexpected_unpublished_records(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _attempt(case)
    partial_root = case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id
    partial_root.mkdir(mode=0o700)
    (partial_root / "readiness.json").write_text("{}")
    (partial_root / "readiness.json").chmod(0o600)

    with pytest.raises(StateCorruptionError, match="unpublished state records"):
        case.writer.load_attempts(attempt.shard_id)


def test_public_attempt_readers_normalize_invalid_and_missing_identities(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)

    with pytest.raises(SlurmStateError, match="invalid shard identity"):
        case.writer.load_attempts(cast(ShardId, "invalid"))
    with pytest.raises(StateNotFoundError, match="shard"):
        case.writer.load_attempts(cast(ShardId, "shard-99999"))
    with pytest.raises(StateConflictError, match="next shard ordinal"):
        case.writer.create_attempt(_attempt(case, ordinal=2))

    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    with pytest.raises(SlurmStateError, match="invalid attempt identity"):
        case.writer.load_attempt(attempt.shard_id, cast(AttemptId, "invalid"))
    with pytest.raises(StateNotFoundError, match="attempt"):
        case.writer.load_attempt(attempt.shard_id, cast(AttemptId, "attempt-9999"))
    with pytest.raises(StateNotFoundError, match="no readiness snapshot"):
        case.writer.load_readiness(attempt.shard_id, attempt.attempt_id)


@pytest.mark.parametrize("collection_root", ("shards", "attempts"))
def test_collection_readers_reject_hidden_unowned_entries(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    collection_root: str,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    root = case.writer.run_root / "shards"
    if collection_root == "attempts":
        root = root / "shard-00000" / "attempts"
    (root / ".unowned").write_text("unexpected")
    (root / ".unowned").chmod(0o600)

    with pytest.raises(StateCorruptionError):
        if collection_root == "shards":
            case.writer.load_shards()
        else:
            case.writer.load_attempts("shard-00000")


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


def test_attempt_update_retry_converges_after_post_replace_fsync_failure(
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
    original_fsync = state_filesystem.os.fsync
    directory_syncs = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 1:
                raise OSError("simulated post-replace fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(state_filesystem.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(SlurmStateError, match="cannot update"):
        case.writer.update_attempt(submitted)

    assert case.writer.update_attempt(submitted) == submitted
    assert directory_syncs == 2
    assert case.writer.load_attempt(attempt.shard_id, attempt.attempt_id) == submitted


def test_concurrent_conflicting_attempt_creation_publishes_one_candidate(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    first = _attempt(case)
    second = _validated_copy(
        first,
        created_at=first.created_at + timedelta(seconds=1),
        updated_at=first.updated_at + timedelta(seconds=1),
    )
    barrier = Barrier(2)

    def create(candidate: AttemptManifest) -> AttemptManifest:
        barrier.wait()
        return SlurmStateWriter(case.workspace, case.plan.run_id).create_attempt(candidate)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = tuple(executor.submit(create, candidate) for candidate in (first, second))

    outcomes = tuple(future.result() for future in futures if future.exception() is None)
    failures = tuple(future.exception() for future in futures if future.exception() is not None)
    assert len(outcomes) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], StateConflictError)
    assert case.writer.load_attempt(first.shard_id, first.attempt_id) == outcomes[0]


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


def test_readiness_reader_rebinds_snapshot_to_its_attempt(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    initial = _readiness(case, attempt)
    case.writer.write_readiness(initial)
    mismatched = _validated_copy(initial, run_id="other-run")
    readiness_path = (
        case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id / "readiness.json"
    )
    readiness_path.write_text(mismatched.serialize_json())

    with pytest.raises(StateCorruptionError, match="invalid persisted readiness"):
        SlurmStateWriter(case.workspace, case.plan.run_id).load_readiness(attempt.shard_id, attempt.attempt_id)


def test_idempotent_readiness_write_revalidates_persisted_snapshot(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    initial = _readiness(case, attempt)
    case.writer.write_readiness(initial)
    mismatched_deployment = _validated_copy(initial.deployments[0], model_alias="other-model")
    mismatched = _validated_copy(initial, deployments=(mismatched_deployment,))
    readiness_path = (
        case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id / "readiness.json"
    )
    readiness_path.write_text(mismatched.serialize_json())

    with pytest.raises(StateCorruptionError, match="invalid persisted readiness"):
        case.writer.write_readiness(mismatched)


def test_readiness_retry_converges_after_post_replace_fsync_failure(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    initial = _readiness(case, attempt)
    case.writer.write_readiness(initial)
    starting = _validated_copy(initial.deployments[0], state=ReadinessState.STARTING)
    revision_two = _validated_copy(
        initial,
        revision=2,
        updated_at=case.created_at + timedelta(minutes=4),
        state=ReadinessState.STARTING,
        deployments=(starting,),
    )
    original_fsync = state_filesystem.os.fsync
    directory_syncs = 0

    def fail_first_directory_fsync(descriptor: int) -> None:
        nonlocal directory_syncs
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            directory_syncs += 1
            if directory_syncs == 1:
                raise OSError("simulated post-replace fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(state_filesystem.os, "fsync", fail_first_directory_fsync)
    with pytest.raises(SlurmStateError, match="cannot persist readiness"):
        case.writer.write_readiness(revision_two)

    assert case.writer.write_readiness(revision_two) == revision_two
    assert directory_syncs == 2
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


def test_candidate_finalization_publishes_one_reloadable_winner_and_seals_the_shard(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)

    winner = case.writer.finalize_winner(
        finalization.attempt.shard_id,
        finalization.attempt.attempt_id,
        published_at=finalization.published_at,
    )

    assert winner.candidate_manifest == finalization.client_result.candidate_output_manifest
    assert case.writer.load_winner(finalization.attempt.shard_id) == winner
    assert stat.S_IMODE((case.writer.run_root / "shards/shard-00000/winner.json").stat().st_mode) == 0o600
    assert (
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at + timedelta(minutes=1),
        )
        == winner
    )
    assert case.writer.create_attempt(finalization.attempt) == finalization.attempt
    with pytest.raises(StateConflictError, match="immutable winner"):
        case.writer.create_attempt(_attempt(case, ordinal=2))
    with pytest.raises(StateConflictError, match="immutable winner"):
        with case.writer.acquire_dataset_workspace(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            "never",
        ):
            pass


def test_finalization_rejects_partial_client_results_and_candidate_file_drift(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    partial = ClientResult.model_validate(
        finalization.client_result.model_dump(mode="python")
        | {
            "actual_records": finalization.client_result.requested_records - 1,
            "outcome": ClientOutcome.PARTIAL,
            "early_shutdown": True,
        }
    )
    _write_state_record(
        _attempt_root(case, finalization.attempt) / "client-result.json",
        partial.serialize_json(),
    )

    with pytest.raises(StateConflictError, match="not eligible"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )

    _write_state_record(
        _attempt_root(case, finalization.attempt) / "client-result.json",
        finalization.client_result.serialize_json(),
    )
    finalization.output_path.write_bytes(b"changed candidate bytes")
    finalization.output_path.chmod(0o600)
    with pytest.raises(StateConflictError, match="not eligible"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )

    with pytest.raises(StateNotFoundError, match="no winner"):
        case.writer.load_winner(finalization.attempt.shard_id)


@pytest.mark.parametrize("file_kind", ("symlink", "hardlink", "fifo", "permissive"))
def test_finalization_rejects_unsafe_candidate_files(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    file_kind: str,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    output_path = finalization.output_path
    if file_kind == "permissive":
        output_path.chmod(0o644)
    else:
        outside = tmp_path / "outside-output"
        outside.write_bytes(output_path.read_bytes())
        outside.chmod(0o600)
        output_path.unlink()
        if file_kind == "symlink":
            output_path.symlink_to(outside)
        elif file_kind == "hardlink":
            os.link(outside, output_path)
        else:
            os.mkfifo(output_path)

    with pytest.raises(StateConflictError, match="not eligible"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )


@pytest.mark.parametrize("record_name", ("client-result.json", "output-manifest.json"))
def test_finalization_classifies_unsafe_result_records_as_corruption(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    record_name: str,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    record_path = _attempt_root(case, finalization.attempt) / record_name
    outside = tmp_path / record_name
    outside.write_text(record_path.read_text())
    outside.chmod(0o600)
    record_path.unlink()
    record_path.symlink_to(outside)

    with pytest.raises(StateCorruptionError, match="unsafe or unreadable"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )


def test_concurrent_identical_finalizers_converge_on_the_same_winner(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    barrier = Barrier(2)

    def finalize() -> ShardWinner:
        barrier.wait()
        writer = SlurmStateWriter(case.workspace, case.plan.run_id)
        return writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        winners = tuple(executor.map(lambda _: finalize(), range(2)))

    assert winners[0] == winners[1] == case.writer.load_winner(finalization.attempt.shard_id)


def test_finalization_retry_converges_after_post_publish_fsync_failure(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    winner_path = case.writer.run_root / "shards/shard-00000/winner.json"
    original_fsync = state_store.os.fsync
    failed = False

    def fail_after_winner_link(descriptor: int) -> None:
        nonlocal failed
        if winner_path.exists() and not failed:
            failed = True
            raise OSError("injected winner directory fsync failure")
        original_fsync(descriptor)

    monkeypatch.setattr(state_store.os, "fsync", fail_after_winner_link)
    with pytest.raises(SlurmStateError, match="cannot finalize"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )

    monkeypatch.setattr(state_store.os, "fsync", original_fsync)
    winner = case.writer.finalize_winner(
        finalization.attempt.shard_id,
        finalization.attempt.attempt_id,
        published_at=finalization.published_at + timedelta(minutes=1),
    )
    assert winner == case.writer.load_winner(finalization.attempt.shard_id)
    assert not tuple(winner_path.parent.glob(".state.*.tmp"))


def test_candidate_file_verification_does_not_hold_the_run_state_lock(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run, multi_node_plan)
    finalization = _complete_finalization_case(case)
    verification_started = Event()
    release_verification = Event()
    original_verify = SlurmStateWriter._verify_candidate_files

    def block_verification(candidate: CandidateOutputManifest) -> None:
        verification_started.set()
        assert release_verification.wait(timeout=5)
        original_verify(candidate)

    monkeypatch.setattr(SlurmStateWriter, "_verify_candidate_files", staticmethod(block_verification))

    with ThreadPoolExecutor(max_workers=1) as executor:
        finalizer = executor.submit(
            case.writer.finalize_winner,
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )
        assert verification_started.wait(timeout=5)
        other_shard = case.shards[1]
        other_attempt = AttemptManifest(
            schema_version=1,
            run_id=case.plan.run_id,
            shard_id=other_shard.shard_id,
            attempt_id="attempt-0001",
            attempt_ordinal=1,
            resolved_plan=case.run.resolved_plan,
            state=AttemptLifecycleState.CREATED,
            created_at=case.created_at + timedelta(minutes=1),
            updated_at=case.created_at + timedelta(minutes=1),
        )
        assert case.writer.create_attempt(other_attempt) == other_attempt
        release_verification.set()
        assert finalizer.result().attempt_id == finalization.attempt.attempt_id


def test_finalization_waits_for_the_dataset_writer_to_release_its_lease(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    finalization_started = Event()
    finalization_finished = Event()

    def finalize(finalization: _FinalizationCase) -> ShardWinner:
        finalization_started.set()
        winner = SlurmStateWriter(case.workspace, case.plan.run_id).finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )
        finalization_finished.set()
        return winner

    with ThreadPoolExecutor(max_workers=1) as executor:
        with case.writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "never") as dataset_path:
            finalization = _persist_complete_result(case, attempt, dataset_path)
            future = executor.submit(finalize, finalization)
            assert finalization_started.wait(timeout=5)
            assert not finalization_finished.wait(timeout=0.1)
        winner = future.result(timeout=5)

    assert winner == case.writer.load_winner(attempt.shard_id)


def test_winner_reader_rejects_a_reference_to_an_unknown_attempt(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    finalization = _complete_finalization_case(case)
    winner = case.writer.finalize_winner(
        finalization.attempt.shard_id,
        finalization.attempt.attempt_id,
        published_at=finalization.published_at,
    )
    corrupted = winner.model_copy(update={"attempt_id": "attempt-9999"})
    _write_state_record(case.writer.run_root / "shards/shard-00000/winner.json", corrupted.serialize_json())

    with pytest.raises(StateCorruptionError, match="unknown attempt"):
        case.writer.load_winner(finalization.attempt.shard_id)
    with pytest.raises(StateCorruptionError, match="unknown attempt"):
        case.writer.finalize_winner(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            published_at=finalization.published_at,
        )
    with pytest.raises(StateCorruptionError, match="unknown attempt"):
        case.writer.create_attempt(_attempt(case, ordinal=2))
    with pytest.raises(StateCorruptionError, match="unknown attempt"):
        with case.writer.acquire_dataset_workspace(
            finalization.attempt.shard_id,
            finalization.attempt.attempt_id,
            "never",
        ):
            pass


def test_resumable_workspace_is_shard_owned_and_exclusively_locked(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    plan_payload = single_node_plan.model_dump(mode="python")
    plan_payload["invocation"]["authored"]["resume"] = "always"
    resumable_plan = ResolvedSlurmRunPlan.model_validate(plan_payload)
    case = _initialized_case(tmp_path, authored_run_single, resumable_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    first_entered = Event()
    release_first = Event()
    second_entered = Event()

    def first_writer() -> Path:
        writer = SlurmStateWriter(case.workspace, case.plan.run_id)
        with writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "always") as path:
            first_entered.set()
            assert release_first.wait(timeout=5)
            return path

    def second_writer() -> Path:
        assert first_entered.wait(timeout=5)
        writer = SlurmStateWriter(case.workspace, case.plan.run_id)
        with writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "always") as path:
            second_entered.set()
            return path

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(first_writer)
        second_future = executor.submit(second_writer)
        assert first_entered.wait(timeout=5)
        assert not second_entered.wait(timeout=0.1)
        release_first.set()
        assert first_future.result() == Path(case.shards[0].resume_workspace.path)
        assert second_future.result() == Path(case.shards[0].resume_workspace.path)

    dataset_path = Path(case.shards[0].resume_workspace.path)
    assert stat.S_IMODE(dataset_path.stat().st_mode) == 0o700
    with pytest.raises(StateConflictError, match="requested dataset workspace"):
        with case.writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "never"):
            pass


def test_dataset_lock_rejects_unsafe_files_without_reclassifying_body_errors(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    lock_path = case.writer.run_root / "shards/shard-00000/resume.lock"
    lock_path.unlink()
    outside = tmp_path / "outside.lock"
    outside.write_text("unchanged")
    lock_path.symlink_to(outside)

    with pytest.raises(SlurmStateError, match="cannot lock dataset workspace"):
        with case.writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "never"):
            pass
    assert outside.read_text() == "unchanged"

    lock_path.unlink()
    with pytest.raises(FileNotFoundError, match="body failure"):
        with case.writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "never"):
            raise FileNotFoundError("body failure")


def test_dataset_lock_interruption_closes_an_open_shard_context(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    shard_closed = False

    @contextmanager
    def open_shard(shard_id: ShardId) -> Iterator[int]:
        nonlocal shard_closed
        del shard_id
        try:
            yield 17
        finally:
            shard_closed = True

    @contextmanager
    def interrupt_lock(directory_descriptor: int, name: str, display_path: Path) -> Iterator[None]:
        del directory_descriptor, name, display_path
        raise KeyboardInterrupt("injected lock interruption")
        yield

    monkeypatch.setattr(case.writer, "_open_shard_directory", open_shard)
    monkeypatch.setattr(state_store, "acquire_file_lock", interrupt_lock)

    with pytest.raises(KeyboardInterrupt, match="lock interruption"):
        with case.writer.acquire_dataset_workspace("shard-00000", "attempt-0001", "never"):
            pass

    assert shard_closed


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


def test_state_package_defers_writer_dependencies_until_access() -> None:
    command = (
        "import sys; "
        "import data_designer.slurm.state as state; "
        "assert 'data_designer.slurm.state.store' not in sys.modules; "
        "assert 'data_designer.slurm.config' not in sys.modules; "
        "assert 'SlurmStateWriter' in dir(state)"
    )

    completed = subprocess.run(
        (sys.executable, "-c", command),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


def test_state_persistence_does_not_import_integration_layer() -> None:
    command = (
        "import sys; "
        "import data_designer.slurm.state.store; "
        "assert 'data_designer.slurm.integration' not in sys.modules"
    )

    completed = subprocess.run(
        (sys.executable, "-c", command),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


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


def _complete_finalization_case(case: _StateCase) -> _FinalizationCase:
    attempt = _submitted_attempt(case)
    case.writer.create_attempt(attempt)
    with case.writer.acquire_dataset_workspace(attempt.shard_id, attempt.attempt_id, "never") as dataset_path:
        return _persist_complete_result(case, attempt, dataset_path)


def _persist_complete_result(case: _StateCase, attempt: AttemptManifest, dataset_path: Path) -> _FinalizationCase:
    output_path = dataset_path / "part-00000.parquet"
    content = b"candidate parquet bytes"
    output_path.write_bytes(content)
    output_path.chmod(0o600)
    requested_records = case.plan.shards[0].requested_records
    candidate = CandidateOutputManifest(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id=attempt.shard_id,
        attempt_id=attempt.attempt_id,
        attempt_ordinal=attempt.attempt_ordinal,
        created_at=case.created_at + timedelta(minutes=3),
        dataset_path=dataset_path.as_posix(),
        requested_records=requested_records,
        actual_records=requested_records,
        outcome=CandidateOutcome.COMPLETE,
        files=(
            CandidateOutputFile(
                relative_path=output_path.name,
                sha256=hashlib.sha256(content).hexdigest(),
                byte_size=len(content),
                record_count=requested_records,
            ),
        ),
        dataset_schema_digest="b" * 64,
        provenance_digest="c" * 64,
    )
    candidate_path = _attempt_root(case, attempt) / "output-manifest.json"
    candidate_reference = ArtifactReference(
        path=candidate_path.as_posix(),
        sha256=candidate.compute_sha256(),
    )
    client_result = ClientResult(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id=attempt.shard_id,
        attempt_id=attempt.attempt_id,
        completed_at=case.created_at + timedelta(minutes=4),
        requested_records=requested_records,
        actual_records=requested_records,
        outcome=ClientOutcome.COMPLETE,
        dataset_path=dataset_path.as_posix(),
        early_shutdown=False,
        requested_resume_mode=case.plan.invocation.authored.resume,
        effective_resume_mode="never",
        candidate_output_manifest=candidate_reference,
    )
    _write_state_record(candidate_path, candidate.serialize_json())
    _write_state_record(_attempt_root(case, attempt) / "client-result.json", client_result.serialize_json())
    completed_attempt = _validated_copy(
        attempt,
        state=AttemptLifecycleState.SUCCEEDED,
        terminal_classification=AttemptTerminalClassification.SUCCEEDED,
        candidate_output=candidate_reference,
        updated_at=case.created_at + timedelta(minutes=5),
    )
    case.writer.update_attempt(completed_attempt)
    return _FinalizationCase(
        attempt=completed_attempt,
        client_result=client_result,
        candidate=candidate,
        published_at=case.created_at + timedelta(minutes=6),
        output_path=output_path,
    )


def _attempt_root(case: _StateCase, attempt: AttemptManifest) -> Path:
    return case.writer.run_root / "shards" / attempt.shard_id / "attempts" / attempt.attempt_id


def _write_state_record(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o600)


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
