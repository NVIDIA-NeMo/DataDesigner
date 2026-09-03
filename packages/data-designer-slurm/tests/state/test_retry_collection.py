# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import Event
from typing import cast

import pytest
from slurm_test_fakes import FakeCommandResponse, FakeSlurmArray, FakeSlurmJob, FakeSlurmRunner, FakeSlurmTask

import data_designer.lazy_heavy_imports as lazy
import data_designer.slurm.state.collection_filesystem as collection_filesystem
import data_designer.slurm.state.collection_merge as collection_merge
import data_designer.slurm.state.collection_storage as collection_storage_module
import data_designer.slurm.state.collection_worker as collection_worker_module
from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.errors import SlurmSubmissionError
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    CollectionState,
    RetryState,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    SlurmCollectionCoordinator,
    SlurmRetryCoordinator,
    SlurmStateError,
    SlurmStateReconciler,
    SlurmStateWriter,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
    compute_candidate_schema_digest,
)
from data_designer.slurm.state.attempt_identity import require_attempt_scheduler_identity
from data_designer.slurm.state.collection_filesystem import derive_collection_staging_directory
from data_designer.slurm.state.collection_storage import CollectionStorage
from data_designer.slurm.state.collection_worker import SlurmCollectionWorker
from data_designer.slurm.state.retry_storage import RetryStorage
from data_designer.slurm.state.storage import StateStorage


@dataclass(frozen=True, slots=True)
class _RunCase:
    workspace: Path
    plan: ResolvedSlurmRunPlan
    run: RunManifest
    shards: tuple[ShardManifest, ...]
    writer: SlurmStateWriter
    created_at: datetime


def test_retry_refreshes_failed_shard_and_publishes_exact_next_attempt(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")

    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)
    attempts = coordinator.retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=5),
    )

    assert len(attempts) == 1
    assert attempts[0].attempt_id == "attempt-0002"
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=4201, array_task_id=0)
    require_attempt_scheduler_identity(
        case.workspace,
        case.plan.run_id,
        attempts[0].shard_id,
        attempts[0].attempt_id,
        SchedulerIdentity(array_job_id=4201, array_task_id=0),
    )
    with pytest.raises(StateConflictError, match="scheduler identity"):
        require_attempt_scheduler_identity(
            case.workspace,
            case.plan.run_id,
            attempts[0].shard_id,
            attempts[0].attempt_id,
            SchedulerIdentity(array_job_id=9999, array_task_id=0),
        )
    assert case.writer.load_attempts(case.shards[0].shard_id) == (attempt, attempts[0])
    assert "#SBATCH --array=0" in cast(str, runner.inputs[-1])
    assert 'DD_ATTEMPT_ORDINAL="0002"' in cast(str, runner.inputs[-1])
    assert (
        coordinator.retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=6),
        )
        == attempts
    )
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 2


def test_retry_rejects_a_nonterminal_explicit_shard(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),)
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    case.writer.create_attempt(
        _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    )

    with pytest.raises(StateConflictError, match="sealed or nonterminal"):
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            shard_ids=(case.shards[0].shard_id,),
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )


def test_retry_rejects_a_shard_with_an_immutable_winner(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),)
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    _publish_all_winners(case)

    with pytest.raises(StateConflictError, match="sealed or nonterminal"):
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            shard_ids=(case.shards[0].shard_id,),
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=7),
        )

    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 1


def test_retry_accepts_unknown_after_bounded_accounting_lag(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state=None)
    SlurmStateReconciler(case.workspace, case.plan.run_id, scheduler).refresh(
        observed_at=case.created_at + timedelta(minutes=3)
    )

    attempts = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=9),
    )

    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=4201, array_task_id=0)


def test_retry_submits_only_the_failed_sparse_array_task(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    initial_tasks = tuple(
        FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=shard.shard_index)) for shard in case.shards
    )
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=initial_tasks),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=1)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    for shard in case.shards:
        case.writer.create_attempt(
            _submitted_attempt(
                case,
                shard,
                scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=shard.shard_index),
            )
        )
    runner.set_task_state(initial_tasks[0].scheduler, queue_state="RUNNING", accounting_state=None)
    runner.set_task_state(initial_tasks[1].scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")

    attempts = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=5),
    )

    assert tuple(attempt.shard_id for attempt in attempts) == ("shard-00001",)
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=4201, array_task_id=1)
    assert "#SBATCH --array=1%2" in cast(str, runner.inputs[-1])
    assert 'case "${DD_ARRAY_TASK_ID}"' in cast(str, runner.inputs[-1])


def test_retry_ambiguous_submission_waits_for_scheduler_visibility_without_a_duplicate(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),)
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    submissions = 0

    def ambiguous_submit(script: str) -> object:
        nonlocal submissions
        del script
        submissions += 1
        raise SlurmSubmissionError("sbatch could not be executed: command timed out", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", ambiguous_submit)
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)

    with pytest.raises(SlurmStateError, match="cannot submit retry"):
        coordinator.retry(effective_resume_mode="never", observed_at=case.created_at + timedelta(minutes=5))

    retry_storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    assert retry_storage.read_status("retry-0001").state is RetryState.PREPARED
    runner.script_next("squeue", FakeCommandResponse())
    runner.script_next("sacct", FakeCommandResponse())
    with pytest.raises(StateConflictError, match="still being reconciled"):
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=6),
        )
    assert submissions == 1


def test_retry_recovers_an_accepted_submission_after_the_receipt_is_lost(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial = FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0))
    retried = FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0))
    runner = FakeSlurmRunner(arrays=(FakeSlurmArray(tasks=(initial,)), FakeSlurmArray(tasks=(retried,))))
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    first_attempt = _submitted_attempt(case, case.shards[0], scheduler=initial.scheduler)
    case.writer.create_attempt(first_attempt)
    runner.set_task_state(initial.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    original_submit = scheduler.submit_script

    def accept_then_lose_receipt(script: str) -> object:
        original_submit(script)
        raise SlurmSubmissionError("sbatch response was lost", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", accept_then_lose_receipt)
    with pytest.raises(SlurmStateError, match="cannot submit retry"):
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )
    monkeypatch.setattr(scheduler, "submit_script", original_submit)
    storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    retry_plan = storage.read_plan("retry-0001")
    runner.script_next(
        "squeue",
        FakeCommandResponse(stdout=f"4201_0|{retry_plan.submission_job_name}\n"),
    )
    runner.script_next("sacct", FakeCommandResponse())

    recovered = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=6),
    )

    assert recovered[0].scheduler == SchedulerIdentity(array_job_id=4201, array_task_id=0)
    assert case.writer.load_attempts(case.shards[0].shard_id) == (first_attempt, recovered[0])
    assert storage.read_status("retry-0001").state is RetryState.COMPLETED
    assert (
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=7),
        )
        == recovered
    )
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 2


@pytest.mark.parametrize(
    ("accepted_before_receipt_loss", "replacement_job_id", "submission_count"),
    [
        pytest.param(False, 4201, 2, id="unaccepted"),
        pytest.param(True, 4301, 3, id="accepted-but-still-invisible"),
    ],
)
def test_retry_replaces_or_fences_an_ambiguous_submission_after_its_deadline(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
    accepted_before_receipt_loss: bool,
    replacement_job_id: int,
    submission_count: int,
) -> None:
    initial = FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0))
    hidden = FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0))
    replacement = FakeSlurmTask(SchedulerIdentity(array_job_id=replacement_job_id, array_task_id=0))
    retry_arrays = (FakeSlurmArray(tasks=(hidden,)),) if accepted_before_receipt_loss else ()
    runner = FakeSlurmRunner(
        arrays=(FakeSlurmArray(tasks=(initial,)), *retry_arrays, FakeSlurmArray(tasks=(replacement,)))
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    first_attempt = _submitted_attempt(case, case.shards[0], scheduler=initial.scheduler)
    case.writer.create_attempt(first_attempt)
    runner.set_task_state(initial.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    original_submit = scheduler.submit_script

    def lose_submission_receipt(script: str) -> object:
        if accepted_before_receipt_loss:
            original_submit(script)
        raise SlurmSubmissionError("sbatch response was lost", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", lose_submission_receipt)
    with pytest.raises(SlurmStateError, match="cannot submit retry"):
        SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )
    monkeypatch.setattr(scheduler, "submit_script", original_submit)
    runner.script_next("squeue", FakeCommandResponse())
    runner.script_next("sacct", FakeCommandResponse())

    attempts = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=11),
    )

    storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("retry-0001").state is RetryState.FAILED
    assert storage.read_status("retry-0002").state is RetryState.COMPLETED
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=replacement_job_id, array_task_id=0)
    with pytest.raises(StateConflictError, match="scheduler identity"):
        require_attempt_scheduler_identity(
            case.workspace,
            case.plan.run_id,
            attempts[0].shard_id,
            attempts[0].attempt_id,
            SchedulerIdentity(array_job_id=4201 if accepted_before_receipt_loss else 4301, array_task_id=0),
        )
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == submission_count


def test_retry_definite_submission_failure_settles_and_can_be_retried(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    runner.script_next("sbatch", FakeCommandResponse(stderr="submission rejected", returncode=2))
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)

    with pytest.raises(SlurmStateError, match="cannot submit retry"):
        coordinator.retry(effective_resume_mode="never", observed_at=case.created_at + timedelta(minutes=5))

    storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("retry-0001").state is RetryState.FAILED
    attempts = coordinator.retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=6),
    )
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=4201, array_task_id=0)
    assert storage.read_status("retry-0002").state is RetryState.COMPLETED


def test_concurrent_retry_requests_converge_on_one_array(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")

    def retry() -> tuple[AttemptManifest, ...]:
        return SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler).retry(
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: retry(), range(2)))

    assert results[0] == results[1]
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 2


def test_default_retry_does_not_hide_failures_outside_an_active_explicit_subset(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    initial_tasks = tuple(
        FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=index)) for index in range(2)
    )
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=initial_tasks),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4301, array_task_id=1)),)),
        )
    )
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    for shard, task in zip(case.shards, initial_tasks, strict=True):
        case.writer.create_attempt(_submitted_attempt(case, shard, scheduler=task.scheduler))
        runner.set_task_state(task.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)

    first = coordinator.retry(
        shard_ids=(case.shards[0].shard_id,),
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=5),
    )
    second = coordinator.retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=6),
    )

    assert tuple(attempt.shard_id for attempt in first) == (case.shards[0].shard_id,)
    assert tuple(attempt.shard_id for attempt in second) == (case.shards[1].shard_id,)
    assert second[0].scheduler == SchedulerIdentity(array_job_id=4301, array_task_id=1)


def test_retry_discards_trailing_journal_interrupted_before_submission(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
        )
    )
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    attempt = _submitted_attempt(case, case.shards[0], scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0))
    case.writer.create_attempt(attempt)
    runner.set_task_state(attempt.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)
    original_publish = RetryStorage.publish_status

    def interrupt_before_status(self: RetryStorage, status: object) -> None:
        del self, status
        raise OSError("injected journal interruption")

    monkeypatch.setattr(RetryStorage, "publish_status", interrupt_before_status)
    with pytest.raises(SlurmStateError, match="cannot retry persisted run"):
        coordinator.retry(effective_resume_mode="never", observed_at=case.created_at + timedelta(minutes=5))
    monkeypatch.setattr(RetryStorage, "publish_status", original_publish)

    attempts = coordinator.retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=6),
    )

    assert attempts[0].attempt_id == "attempt-0002"
    assert RetryStorage(StateStorage(case.workspace, case.plan.run_id)).list_retry_ids() == ("retry-0001",)


def test_retry_recovers_evolved_attempt_and_its_exact_winner(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run_single, single_node_plan)
    initial = FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=0))
    retried = FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0))
    runner = FakeSlurmRunner(arrays=(FakeSlurmArray(tasks=(initial,)), FakeSlurmArray(tasks=(retried,))))
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    first_attempt = _submitted_attempt(case, case.shards[0], scheduler=initial.scheduler)
    case.writer.create_attempt(first_attempt)
    runner.set_task_state(initial.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)
    original_complete = SlurmRetryCoordinator._complete_retry

    def interrupt_after_attempts(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected completion interruption")

    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", interrupt_after_attempts)
    with pytest.raises(SlurmStateError, match="cannot retry persisted run"):
        coordinator.retry(effective_resume_mode="never", observed_at=case.created_at + timedelta(minutes=5))
    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", original_complete)
    retry_attempt = case.writer.load_attempts(case.shards[0].shard_id)[-1]
    running = _copy_attempt(
        retry_attempt,
        state=AttemptLifecycleState.RUNNING,
        updated_at=case.created_at + timedelta(minutes=6),
    )
    case.writer.update_attempt(running)
    with case.writer.acquire_dataset_workspace(case.shards[0].shard_id, running.attempt_id, "never") as dataset_path:
        _publish_candidate(case, case.shards[0], running, dataset_path)
    succeeded = case.writer.load_attempts(case.shards[0].shard_id)[-1]
    case.writer.finalize_winner(
        case.shards[0].shard_id,
        succeeded.attempt_id,
        published_at=case.created_at + timedelta(minutes=10),
    )
    runner.set_task_state(retried.scheduler, queue_state=None, accounting_state="COMPLETED")

    recovered = coordinator.retry(
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=11),
    )

    assert recovered == (succeeded,)
    assert (
        RetryStorage(StateStorage(case.workspace, case.plan.run_id)).read_status("retry-0001").state
        is RetryState.COMPLETED
    )


def test_retry_settles_submitted_journal_before_serving_a_disjoint_selection(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_tasks = tuple(
        FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=index)) for index in range(2)
    )
    runner = FakeSlurmRunner(
        arrays=(
            FakeSlurmArray(tasks=initial_tasks),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0)),)),
            FakeSlurmArray(tasks=(FakeSlurmTask(SchedulerIdentity(array_job_id=4301, array_task_id=1)),)),
        )
    )
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    for shard, task in zip(case.shards, initial_tasks, strict=True):
        case.writer.create_attempt(_submitted_attempt(case, shard, scheduler=task.scheduler))
        runner.set_task_state(task.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)
    original_complete = SlurmRetryCoordinator._complete_retry

    def interrupt_after_attempts(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected completion interruption")

    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", interrupt_after_attempts)
    with pytest.raises(SlurmStateError, match="cannot retry persisted run"):
        coordinator.retry(
            shard_ids=(case.shards[0].shard_id,),
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )
    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", original_complete)

    attempts = coordinator.retry(
        shard_ids=(case.shards[1].shard_id,),
        effective_resume_mode="never",
        observed_at=case.created_at + timedelta(minutes=6),
    )

    assert tuple(attempt.shard_id for attempt in attempts) == (case.shards[1].shard_id,)
    assert attempts[0].scheduler == SchedulerIdentity(array_job_id=4301, array_task_id=1)
    storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("retry-0001").state is RetryState.COMPLETED
    assert storage.read_status("retry-0002").state is RetryState.COMPLETED


def test_retry_does_not_return_recovered_attempt_for_a_different_resume_mode(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    initial_tasks = tuple(
        FakeSlurmTask(SchedulerIdentity(array_job_id=4101, array_task_id=index)) for index in range(2)
    )
    retried = FakeSlurmTask(SchedulerIdentity(array_job_id=4201, array_task_id=0))
    runner = FakeSlurmRunner(arrays=(FakeSlurmArray(tasks=initial_tasks), FakeSlurmArray(tasks=(retried,))))
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    scheduler = SlurmCommandClient(runner)
    scheduler.submit_script("initial")
    for shard, task in zip(case.shards, initial_tasks, strict=True):
        case.writer.create_attempt(_submitted_attempt(case, shard, scheduler=task.scheduler))
        runner.set_task_state(task.scheduler, queue_state=None, accounting_state="FAILED", exit_code="1:0")
    coordinator = SlurmRetryCoordinator(case.workspace, case.plan.run_id, scheduler)
    original_complete = SlurmRetryCoordinator._complete_retry

    def interrupt_after_attempts(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected completion interruption")

    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", interrupt_after_attempts)
    with pytest.raises(SlurmStateError, match="cannot retry persisted run"):
        coordinator.retry(
            shard_ids=(case.shards[0].shard_id,),
            effective_resume_mode="never",
            observed_at=case.created_at + timedelta(minutes=5),
        )
    monkeypatch.setattr(SlurmRetryCoordinator, "_complete_retry", original_complete)

    with pytest.raises(StateConflictError, match="sealed or nonterminal"):
        coordinator.retry(
            shard_ids=(case.shards[0].shard_id,),
            effective_resume_mode="always",
            observed_at=case.created_at + timedelta(minutes=6),
        )

    storage = RetryStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("retry-0001").state is RetryState.COMPLETED
    assert storage.list_retry_ids() == ("retry-0001",)


def test_collection_submits_cpu_job_and_publishes_ordered_winners_atomically(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))

    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))

    assert submitted.state is CollectionState.SUBMITTED
    assert submitted.scheduler == 5101
    script = cast(str, runner.inputs[-1])
    assert "data_designer.slurm.state.collection_worker" in script
    assert "--gpus" not in script
    assert "--gres" not in script
    stale_stage = Path(case.plan.output.root).parent / submitted.staging_directory
    stale_stage.mkdir(mode=0o700)
    (stale_stage / "partial").write_text("incomplete")
    unrelated_stage = Path(case.plan.output.root).parent / f".dd-collection-{'f' * 32}.tmp"
    unrelated_stage.mkdir(mode=0o700)
    (unrelated_stage / "active").write_text("preserve")
    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))

    destination = Path(case.plan.output.root)
    assert result.actual_records == case.plan.invocation.authored.num_records
    assert len(result.files) == case.plan.output.partitions
    assert lazy.pq.read_table(destination / result.files[0].relative_path).column("record_id").to_pylist() == list(
        range(case.plan.invocation.authored.num_records)
    )
    persisted = CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).read_status(submitted.collection_id)
    assert persisted.state is CollectionState.SUCCEEDED
    assert coordinator.submit() == persisted
    assert not stale_stage.exists()
    assert (unrelated_stage / "active").read_text() == "preserve"


def test_collection_requires_every_planned_winner_before_submission(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)

    with pytest.raises(StateNotFoundError, match="has no winner"):
        SlurmCollectionCoordinator(case.workspace, case.plan.run_id).submit(
            submitted_at=case.created_at + timedelta(minutes=10)
        )


def test_collection_prepares_missing_authorized_destination_parents(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    workspace_root = Path(multi_node_plan.selected_profile.profile.workspace_root)
    output = multi_node_plan.output.model_copy(
        update={"root": (workspace_root / "new" / "nested" / "output").as_posix()}
    )
    plan = ResolvedSlurmRunPlan.model_validate_json(
        json.dumps(multi_node_plan.model_copy(update={"output": output}).model_dump(mode="json"))
    )
    case = _initialize_run(tmp_path, authored_run, plan)
    destination = Path(case.plan.output.root)
    assert not destination.parent.exists()
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))

    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    assert destination.parent.is_dir()
    assert not destination.exists()
    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    assert result.actual_records == case.plan.invocation.authored.num_records
    assert destination.is_dir()


def test_concurrent_collection_submission_converges_on_one_job(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))

    def submit() -> object:
        return SlurmCollectionCoordinator(
            case.workspace,
            case.plan.run_id,
            SlurmCommandClient(runner),
        ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(lambda _: submit(), range(2)))

    assert results[0] == results[1]
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 1


def test_collection_ambiguous_submission_waits_for_scheduler_visibility_without_a_duplicate(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner()
    scheduler = SlurmCommandClient(runner)
    submissions = 0

    def ambiguous_submit(script: str) -> object:
        nonlocal submissions
        del script
        submissions += 1
        raise SlurmSubmissionError("sbatch could not be executed: command timed out", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", ambiguous_submit)
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler)

    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))

    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("collection-0001").state is CollectionState.PREPARED
    runner.script_next("squeue", FakeCommandResponse())
    runner.script_next("sacct", FakeCommandResponse())
    with pytest.raises(StateConflictError, match="still being reconciled"):
        SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).submit(
            submitted_at=case.created_at + timedelta(minutes=11)
        )
    assert submissions == 1


def test_collection_recovers_an_accepted_submission_after_the_receipt_is_lost(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    scheduler = SlurmCommandClient(runner)
    original_submit = scheduler.submit_script

    def accept_then_lose_receipt(script: str) -> object:
        original_submit(script)
        raise SlurmSubmissionError("sbatch response was lost", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", accept_then_lose_receipt)
    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).submit(
            submitted_at=case.created_at + timedelta(minutes=10)
        )
    monkeypatch.setattr(scheduler, "submit_script", original_submit)
    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    collection_plan = storage.read_plan("collection-0001")
    runner.script_next(
        "squeue",
        FakeCommandResponse(stdout=f"5101|{collection_plan.submission_job_name}\n"),
    )
    runner.script_next("sacct", FakeCommandResponse())
    waiting_for_binding = Event()
    binding_published = Event()

    def wait_for_binding(seconds: float) -> None:
        assert seconds == 1
        waiting_for_binding.set()
        assert binding_published.wait(timeout=2)

    monkeypatch.setattr(collection_worker_module, "sleep", wait_for_binding)

    with ThreadPoolExecutor(max_workers=1) as executor:
        worker_result = executor.submit(
            SlurmCollectionWorker(
                case.workspace,
                case.plan.run_id,
                "collection-0001",
                environment={"SLURM_JOB_ID": "5101"},
            ).run,
            completed_at=case.created_at + timedelta(minutes=12),
        )
        assert waiting_for_binding.wait(timeout=2)
        recovered = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).refresh(
            observed_at=case.created_at + timedelta(minutes=11)
        )
        binding_published.set()
        result = worker_result.result(timeout=5)

    assert recovered.collection_id == "collection-0001"
    assert recovered.state is CollectionState.PENDING
    assert recovered.scheduler == 5101
    assert result.actual_records == case.plan.invocation.authored.num_records
    settled = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).submit(
        submitted_at=case.created_at + timedelta(minutes=13)
    )
    assert settled.state is CollectionState.SUCCEEDED
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 1


def test_collection_fences_an_invisible_accepted_submission_before_replacement_writes(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101), FakeSlurmJob(5102)))
    scheduler = SlurmCommandClient(runner)
    original_submit = scheduler.submit_script

    def accept_then_lose_receipt(script: str) -> object:
        original_submit(script)
        raise SlurmSubmissionError("sbatch response was lost", may_have_succeeded=True)

    monkeypatch.setattr(scheduler, "submit_script", accept_then_lose_receipt)
    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).submit(
            submitted_at=case.created_at + timedelta(minutes=10)
        )
    monkeypatch.setattr(scheduler, "submit_script", original_submit)
    runner.script_next("squeue", FakeCommandResponse())
    runner.script_next("sacct", FakeCommandResponse())

    submitted = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, scheduler).submit(
        submitted_at=case.created_at + timedelta(minutes=16)
    )

    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("collection-0001").state is CollectionState.FAILED
    assert submitted.collection_id == "collection-0002"
    assert submitted.state is CollectionState.SUBMITTED
    assert submitted.scheduler == 5102
    with pytest.raises(StateConflictError, match="ordinary Slurm job identity"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            "collection-0001",
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=17))
    assert not Path(case.plan.output.root).exists()
    assert sum(Path(call[0]).name == "sbatch" for call in runner.calls) == 2


def test_collection_definite_submission_failure_settles_and_can_be_retried(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    runner.script_next("sbatch", FakeCommandResponse(stderr="submission rejected", returncode=2))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))

    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))

    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    assert storage.read_status("collection-0001").state is CollectionState.FAILED
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=11))
    assert submitted.collection_id == "collection-0002"
    assert submitted.state is CollectionState.SUBMITTED


def test_collection_discards_trailing_journal_interrupted_before_submission(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    original_publish = CollectionStorage.publish_status

    def interrupt_before_status(self: CollectionStorage, status: object) -> None:
        del self, status
        raise OSError("injected journal interruption")

    monkeypatch.setattr(CollectionStorage, "publish_status", interrupt_before_status)
    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    monkeypatch.setattr(CollectionStorage, "publish_status", original_publish)

    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=11))

    assert submitted.collection_id == "collection-0001"
    assert CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).list_collection_ids() == (
        "collection-0001",
    )


def test_collection_failure_removes_staging_without_publishing_partial_output(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    def fail_after_writes(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise OSError("injected collection failure")

    monkeypatch.setattr(
        "data_designer.slurm.state.collection_merge.CollectionMerger._write_result",
        fail_after_writes,
    )
    with pytest.raises(SlurmStateError, match="collection .* failed"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=11))

    destination = Path(case.plan.output.root)
    assert not destination.exists()
    assert not tuple(destination.parent.glob(".dd-*.tmp"))
    persisted = CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).read_status(submitted.collection_id)
    assert persisted.state is CollectionState.FAILED


def test_collection_rejects_output_replacement_before_descriptor_digest(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    original_describe = collection_merge._BoundOutput.describe
    injected = False

    def replace_output_before_describe(
        output: collection_merge._BoundOutput,
        relative_path: str,
        record_count: int,
    ) -> object:
        nonlocal injected
        if not injected:
            injected = True
            output.path.rename(output.path.with_suffix(f"{output.path.suffix}.written"))
            output.path.write_bytes(b"replacement")
            output.path.chmod(0o600)
        return original_describe(output, relative_path, record_count)

    monkeypatch.setattr(collection_merge._BoundOutput, "describe", replace_output_before_describe)

    with pytest.raises(SlurmStateError, match="collection .* failed"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=11))

    destination = Path(case.plan.output.root)
    assert not destination.exists()
    assert not (destination.parent / submitted.staging_directory).exists()
    persisted = CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).read_status(submitted.collection_id)
    assert persisted.state is CollectionState.FAILED


def test_collection_refresh_cleans_only_its_exact_stage_after_oom(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    parent = Path(case.plan.output.root).parent
    exact_stage = parent / submitted.staging_directory
    exact_stage.mkdir(mode=0o700)
    (exact_stage / "partial.parquet").write_text("incomplete")
    unrelated_stage = parent / f".dd-collection-{'e' * 32}.tmp"
    unrelated_stage.mkdir(mode=0o700)
    (unrelated_stage / "active").write_text("preserve")
    runner.set_job_state(5101, queue_state=None, accounting_state="OUT_OF_MEMORY", exit_code="0:125")

    refreshed = coordinator.refresh(observed_at=case.created_at + timedelta(minutes=11))

    assert refreshed.state is CollectionState.FAILED
    assert not exact_stage.exists()
    assert (unrelated_stage / "active").read_text() == "preserve"
    assert not Path(case.plan.output.root).exists()


def test_collection_worker_reauthorizes_persisted_partition_intent(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    plan = storage.read_plan(submitted.collection_id)
    tampered = plan.model_copy(update={"num_partitions": plan.num_partitions + 1})
    storage.get_plan_path(plan.collection_id).write_text(tampered.serialize_json())
    storage.replace_status(
        submitted.model_copy(
            update={
                "collection_plan": storage.get_plan_reference(tampered),
                "staging_directory": derive_collection_staging_directory(tampered),
            }
        )
    )

    with pytest.raises(StateConflictError, match="partition count"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=11))

    assert not Path(case.plan.output.root).exists()


def test_succeeded_collection_rejects_modified_partition_bytes(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    output_path = Path(case.plan.output.root) / result.files[0].relative_path
    output_path.write_bytes(output_path.read_bytes() + b"tampered")

    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        coordinator.submit()


def test_succeeded_collection_rejects_same_size_partition_mutation(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    output_path = Path(case.plan.output.root) / result.files[0].relative_path
    content = bytearray(output_path.read_bytes())
    content[-1] ^= 1
    output_path.write_bytes(content)

    with pytest.raises(SlurmStateError, match="cannot submit collection"):
        coordinator.submit()


def test_succeeded_collection_status_check_does_not_read_partition_payloads(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    original_open = collection_storage_module.open_verified_regular_file

    def reject_partition_reads(*args: object, **kwargs: object) -> object:
        if args[1] != "collection-result.json":
            raise AssertionError("login-host validation attempted to read collected partition bytes")
        return original_open(*args, **kwargs)

    monkeypatch.setattr(collection_storage_module, "open_verified_regular_file", reject_partition_reads)

    assert coordinator.submit().state is CollectionState.SUCCEEDED


def test_succeeded_collection_rejects_status_result_digest_mismatch(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    coordinator = SlurmCollectionCoordinator(case.workspace, case.plan.run_id, SlurmCommandClient(runner))
    submitted = coordinator.submit(submitted_at=case.created_at + timedelta(minutes=10))
    SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    succeeded = storage.read_status(submitted.collection_id)
    assert succeeded.result is not None
    storage.replace_status(
        succeeded.model_copy(
            update={
                "result": succeeded.result.model_copy(update={"sha256": "f" * 64}),
            }
        )
    )

    with pytest.raises(StateCorruptionError, match="does not bind its published result"):
        coordinator.submit()


def test_collection_refuses_destination_collision_before_submission(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    destination = Path(case.plan.output.root)
    destination.mkdir(mode=0o700)
    marker = destination / "existing.txt"
    marker.write_text("preserve")
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))

    with pytest.raises(StateConflictError, match="already exists"):
        SlurmCollectionCoordinator(
            case.workspace,
            case.plan.run_id,
            SlurmCommandClient(runner),
        ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    assert marker.read_text() == "preserve"
    assert not any(Path(call[0]).name == "sbatch" for call in runner.calls)


def test_collection_refuses_atomic_publication_collision_without_replacement(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    destination = Path(case.plan.output.root)
    original_rename = collection_filesystem._rename_without_overwrite

    def collide(
        source_directory: int,
        source_name: str,
        destination_directory: int,
        destination_name: str,
    ) -> None:
        destination.mkdir(mode=0o700)
        (destination / "existing.txt").write_text("preserve")
        original_rename(source_directory, source_name, destination_directory, destination_name)

    monkeypatch.setattr(collection_filesystem, "_rename_without_overwrite", collide)

    with pytest.raises(StateConflictError, match="already exists"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=11))

    assert (destination / "existing.txt").read_text() == "preserve"
    assert not (destination.parent / submitted.staging_directory).exists()


def test_collection_detects_destination_parent_replacement_before_success(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace_root = Path(multi_node_plan.selected_profile.profile.workspace_root)
    output = multi_node_plan.output.model_copy(update={"root": (workspace_root / "exports" / "output").as_posix()})
    plan = ResolvedSlurmRunPlan.model_validate_json(
        json.dumps(multi_node_plan.model_copy(update={"output": output}).model_dump(mode="json"))
    )
    case = _initialize_run(tmp_path, authored_run, plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    destination = Path(case.plan.output.root)
    original_parent = destination.parent
    moved_parent = original_parent.with_name("moved-exports")
    original_rename = collection_filesystem._rename_without_overwrite
    injected = False

    def replace_parent(
        source_directory: int,
        source_name: str,
        destination_directory: int,
        destination_name: str,
    ) -> None:
        nonlocal injected
        if not injected:
            injected = True
            original_parent.rename(moved_parent)
            original_parent.mkdir(mode=0o700)
        original_rename(source_directory, source_name, destination_directory, destination_name)

    monkeypatch.setattr(collection_filesystem, "_rename_without_overwrite", replace_parent)

    with pytest.raises(SlurmStateError, match="collection .* failed"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={"SLURM_JOB_ID": "5101"},
        ).run(completed_at=case.created_at + timedelta(minutes=11))

    assert not destination.exists()
    assert not (moved_parent / destination.name).exists()
    assert not (moved_parent / submitted.staging_directory).exists()
    persisted = CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).read_status(submitted.collection_id)
    assert persisted.state is CollectionState.FAILED


def test_collection_recovers_when_publication_completed_before_reported_error(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    original_rename = collection_filesystem._rename_without_overwrite

    def publish_then_fail(
        source_directory: int,
        source_name: str,
        destination_directory: int,
        destination_name: str,
    ) -> None:
        original_rename(source_directory, source_name, destination_directory, destination_name)
        raise OSError("injected post-rename failure")

    monkeypatch.setattr(collection_filesystem, "_rename_without_overwrite", publish_then_fail)

    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))

    assert result.actual_records == case.plan.invocation.authored.num_records
    persisted = CollectionStorage(StateStorage(case.workspace, case.plan.run_id)).read_status(submitted.collection_id)
    assert persisted.state is CollectionState.SUCCEEDED
    assert Path(case.plan.output.root).is_dir()


def test_collection_worker_rejects_login_node_execution(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    with pytest.raises(StateConflictError, match="inside its recorded Slurm job"):
        SlurmCollectionWorker(
            case.workspace,
            case.plan.run_id,
            submitted.collection_id,
            environment={},
        ).run(completed_at=case.created_at + timedelta(minutes=11))


def test_collection_worker_rejects_login_node_recovery_before_reading_outputs(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case = _initialize_run(tmp_path, authored_run, multi_node_plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))
    SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))
    storage = CollectionStorage(StateStorage(case.workspace, case.plan.run_id))
    storage.replace_status(submitted)

    def reject_output_reads(*args: object, **kwargs: object) -> None:
        del args, kwargs
        raise AssertionError("login-host worker attempted to read collected outputs")

    monkeypatch.setattr(CollectionStorage, "verify_result_files", reject_output_reads)
    worker = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={},
    )

    with pytest.raises(StateConflictError, match="inside its recorded Slurm job"):
        worker.run(completed_at=case.created_at + timedelta(minutes=12))

    assert storage.read_status(submitted.collection_id) == submitted


def test_collection_worker_validates_location_before_constructing_state(tmp_path: Path) -> None:
    with pytest.raises(SlurmStateError, match="invalid persisted collection worker location"):
        SlurmCollectionWorker("relative/workspace", "run-001", "collection-0001")
    with pytest.raises(SlurmStateError, match="invalid persisted collection worker location"):
        SlurmCollectionWorker(tmp_path, "../run", "collection-0001")


@pytest.mark.parametrize("output_format", ["csv", "jsonl"])
def test_collection_exports_non_parquet_formats_in_deterministic_partitions(
    tmp_path: Path,
    authored_run: DataDesignerSlurmConfig,
    multi_node_plan: ResolvedSlurmRunPlan,
    output_format: str,
) -> None:
    output = multi_node_plan.output.model_copy(update={"format": output_format, "partitions": 3})
    plan = ResolvedSlurmRunPlan.model_validate_json(
        json.dumps(multi_node_plan.model_copy(update={"output": output}).model_dump(mode="json"))
    )
    case = _initialize_run(tmp_path, authored_run, plan)
    _publish_all_winners(case)
    runner = FakeSlurmRunner(jobs=(FakeSlurmJob(5101),))
    submitted = SlurmCollectionCoordinator(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(runner),
    ).submit(submitted_at=case.created_at + timedelta(minutes=10))

    result = SlurmCollectionWorker(
        case.workspace,
        case.plan.run_id,
        submitted.collection_id,
        environment={"SLURM_JOB_ID": "5101"},
    ).run(completed_at=case.created_at + timedelta(minutes=11))

    assert len(result.files) == 3
    requested_records = case.plan.invocation.authored.num_records
    floor_count = requested_records // 3
    assert tuple(file.record_count for file in result.files) == (
        floor_count,
        floor_count,
        requested_records - 2 * floor_count,
    )
    values: list[int] = []
    for output_file in result.files:
        path = Path(case.plan.output.root) / output_file.relative_path
        if output_format == "csv":
            with path.open(newline="") as source:
                values.extend(int(row["record_id"]) for row in csv.DictReader(source))
        else:
            values.extend(int(json.loads(line)["record_id"]) for line in path.read_text().splitlines())
    assert values == list(range(requested_records))


def _initialize_run(
    tmp_path: Path,
    authored_config: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
) -> _RunCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    relocated_plan = _relocate_plan(plan, workspace)
    created_at = datetime(2026, 9, 1, 12, tzinfo=timezone.utc)
    run_root = workspace / "runs" / relocated_plan.run_id
    run = RunManifest(
        schema_version=1,
        run_id=relocated_plan.run_id,
        created_at=created_at,
        authored_config=relocated_plan.authored_config,
        resolved_plan=ArtifactReference(
            path=(run_root / "resolved-plan.json").as_posix(),
            sha256=relocated_plan.compute_sha256(),
        ),
        shard_count=len(relocated_plan.shards),
    )
    shards = tuple(
        ShardManifest(
            schema_version=1,
            run_id=run.run_id,
            shard_id=planned.shard_id,
            shard_index=planned.shard_index,
            record_range=planned.record_range,
            input_partition=planned.input_partition,
            resume_workspace=planned.resume_workspace,
            created_at=created_at,
        )
        for planned in relocated_plan.shards
    )
    writer = SlurmStateWriter(workspace, run.run_id)
    writer.initialize_run(authored_config, relocated_plan, run, shards)
    return _RunCase(workspace, relocated_plan, run, shards, writer, created_at)


def _relocate_plan(plan: ResolvedSlurmRunPlan, workspace: Path) -> ResolvedSlurmRunPlan:
    previous_workspace = plan.selected_profile.profile.workspace_root
    payload = cast(
        dict[str, object],
        json.loads(plan.serialize_json().replace(previous_workspace, workspace.as_posix())),
    )
    selected_profile = cast(dict[str, object], payload["selected_profile"])
    profile_payload = cast(dict[str, object], selected_profile["profile"])
    profile_mounts = cast(list[dict[str, object]], profile_payload["container_mounts"])
    resolved_mounts = cast(list[dict[str, object]], payload["container_mounts"])
    for mount in (*profile_mounts, *resolved_mounts):
        mount["source"] = workspace.parent.as_posix()
        mount["target"] = workspace.parent.as_posix()
    profile = SlurmProfile.model_validate(selected_profile["profile"])
    selected_profile["profile_sha256"] = compute_canonical_json_sha256(profile.model_dump(mode="json"))
    return ResolvedSlurmRunPlan.model_validate_json(json.dumps(payload))


def _submitted_attempt(
    case: _RunCase,
    shard: ShardManifest,
    *,
    scheduler: SchedulerIdentity,
) -> AttemptManifest:
    return AttemptManifest(
        schema_version=1,
        run_id=case.run.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=case.run.resolved_plan,
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=scheduler,
        created_at=case.created_at + timedelta(minutes=1),
        updated_at=case.created_at + timedelta(minutes=1),
    )


def _publish_all_winners(case: _RunCase) -> None:
    for shard in case.shards:
        scheduler = SchedulerIdentity(array_job_id=4101, array_task_id=shard.shard_index)
        submitted = _submitted_attempt(case, shard, scheduler=scheduler)
        case.writer.create_attempt(submitted)
        running = _copy_attempt(
            submitted,
            state=AttemptLifecycleState.RUNNING,
            updated_at=case.created_at + timedelta(minutes=2),
        )
        case.writer.update_attempt(running)
        with case.writer.acquire_dataset_workspace(shard.shard_id, submitted.attempt_id, "never") as dataset_path:
            _publish_candidate(case, shard, running, dataset_path)
        case.writer.finalize_winner(
            shard.shard_id,
            running.attempt_id,
            published_at=case.created_at + timedelta(minutes=6),
        )


def _publish_candidate(case: _RunCase, shard: ShardManifest, running: AttemptManifest, dataset_path: Path) -> None:
    output_path = dataset_path / "part-00000.parquet"
    values = range(shard.record_range.start_index, shard.record_range.end_index_exclusive)
    table = lazy.pa.table({"record_id": values})
    lazy.pq.write_table(table, output_path)
    output_path.chmod(0o644)
    content = output_path.read_bytes()
    candidate_path = (
        case.writer.run_root / "shards" / shard.shard_id / "attempts" / running.attempt_id / "output-manifest.json"
    )
    candidate = CandidateOutputManifest(
        schema_version=1,
        run_id=case.run.run_id,
        shard_id=shard.shard_id,
        attempt_id=running.attempt_id,
        attempt_ordinal=running.attempt_ordinal,
        created_at=running.updated_at + timedelta(minutes=1),
        dataset_path=dataset_path.as_posix(),
        requested_records=shard.record_range.record_count,
        actual_records=shard.record_range.record_count,
        outcome=CandidateOutcome.COMPLETE,
        files=(
            CandidateOutputFile(
                relative_path=output_path.name,
                sha256=hashlib.sha256(content).hexdigest(),
                byte_size=len(content),
                record_count=shard.record_range.record_count,
            ),
        ),
        dataset_schema_digest=compute_candidate_schema_digest(table.schema),
        provenance_digest=case.plan.compute_sha256(),
    )
    candidate_reference = ArtifactReference(path=candidate_path.as_posix(), sha256=candidate.compute_sha256())
    result = ClientResult(
        schema_version=1,
        run_id=case.run.run_id,
        shard_id=shard.shard_id,
        attempt_id=running.attempt_id,
        completed_at=running.updated_at + timedelta(minutes=2),
        requested_records=shard.record_range.record_count,
        actual_records=shard.record_range.record_count,
        outcome=ClientOutcome.COMPLETE,
        dataset_path=dataset_path.as_posix(),
        early_shutdown=False,
        requested_resume_mode=case.plan.invocation.authored.resume,
        effective_resume_mode="never",
        candidate_output_manifest=candidate_reference,
    )
    case.writer.publish_attempt_result(result, candidate)
    completed = _copy_attempt(
        running,
        state=AttemptLifecycleState.SUCCEEDED,
        terminal_classification=AttemptTerminalClassification.SUCCEEDED,
        candidate_output=candidate_reference,
        updated_at=running.updated_at + timedelta(minutes=3),
    )
    case.writer.update_attempt(completed)


def _copy_attempt(attempt: AttemptManifest, **updates: object) -> AttemptManifest:
    payload = attempt.model_dump(mode="python")
    payload.update(updates)
    return AttemptManifest.model_validate(payload)
