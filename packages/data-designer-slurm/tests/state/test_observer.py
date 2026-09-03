# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import stat
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import cast

import pytest
from slurm_test_fakes import FakeSlurmRunner

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig, SlurmProfile
from data_designer.slurm.contracts import ArtifactReference, compute_canonical_json_sha256
from data_designer.slurm.launcher.client import SlurmCommandClient
from data_designer.slurm.launcher.models import SlurmAccountingEntry, SlurmProcessExitCode, SlurmQueueEntry
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    CandidateOutcome,
    CandidateOutputFile,
    CandidateOutputManifest,
    EffectiveAttemptState,
    EffectiveRunState,
    GenerationState,
    RunManifest,
    SchedulerIdentity,
    SchedulerJobIdentity,
    SchedulerObservation,
    SchedulerObservationCollector,
    SchedulerState,
    ShardManifest,
    ShardWinner,
    SlurmStateError,
    SlurmStateReconciler,
    SlurmStateWriter,
    StateConflictError,
    StateCorruptionError,
)


@dataclass(frozen=True, slots=True)
class _ReconciliationCase:
    workspace: Path
    plan: ResolvedSlurmRunPlan
    run: RunManifest
    shard: ShardManifest
    attempt: AttemptManifest
    writer: SlurmStateWriter
    created_at: datetime


@dataclass(frozen=True, slots=True)
class _StaticSchedulerClient:
    queue: tuple[SlurmQueueEntry, ...]
    accounting: tuple[SlurmAccountingEntry, ...]

    def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
        del selectors
        return self.queue

    def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
        del selectors
        return self.accounting


def test_collector_prefers_terminal_accounting_for_array_and_collection_jobs() -> None:
    task = SchedulerIdentity(array_job_id=4101, array_task_id=0)
    queue = (
        SlurmQueueEntry(job_identity=task, state=SchedulerState.RUNNING),
        SlurmQueueEntry(job_identity=5101, state=SchedulerState.RUNNING),
    )
    accounting = (
        _accounting(task, SchedulerState.NODE_FAILED),
        _accounting(5101, SchedulerState.COMPLETED),
    )
    observed_at = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)

    observations = SchedulerObservationCollector(_StaticSchedulerClient(queue, accounting)).collect(
        (task, 5101),
        observed_at=observed_at,
    )

    assert tuple(observation.state for observation in observations) == (
        SchedulerState.NODE_FAILED,
        SchedulerState.COMPLETED,
    )
    assert tuple(observation.scheduler for observation in observations) == (task, 5101)


def test_fresh_process_refresh_persists_one_fixed_accounting_lag_deadline(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    SlurmCommandClient(fake_slurm_runner).submit("run.sbatch")
    scheduler = cast(SchedulerIdentity, case.attempt.scheduler)
    fake_slurm_runner.set_task_state(scheduler, queue_state=None, accounting_state=None)
    first_time = case.created_at + timedelta(minutes=3)

    first = SlurmStateReconciler(case.workspace, case.plan.run_id, SlurmCommandClient(fake_slurm_runner)).refresh(
        observed_at=first_time
    )
    second = SlurmStateReconciler(case.workspace, case.plan.run_id, SlurmCommandClient(fake_slurm_runner)).refresh(
        observed_at=first_time + timedelta(minutes=1)
    )
    expired = SlurmStateReconciler(case.workspace, case.plan.run_id, SlurmCommandClient(fake_slurm_runner)).refresh(
        observed_at=first_time + timedelta(minutes=6)
    )

    first_scheduler = first.shards[0].attempts[0].scheduler
    second_scheduler = second.shards[0].attempts[0].scheduler
    assert first.effective_state is EffectiveRunState.ACCOUNTING_LAG
    assert second.effective_state is EffectiveRunState.ACCOUNTING_LAG
    assert expired.effective_state is EffectiveRunState.UNKNOWN
    assert first_scheduler is not None and second_scheduler is not None
    assert first_scheduler.reconciliation_deadline == second_scheduler.reconciliation_deadline
    scheduler_path = case.writer.run_root / "shards/shard-00000/attempts/attempt-0001/scheduler.json"
    assert stat.S_IMODE(scheduler_path.stat().st_mode) == 0o600


def test_refresh_uses_terminal_accounting_over_stale_active_queue_state(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    SlurmCommandClient(fake_slurm_runner).submit("run.sbatch")
    scheduler = cast(SchedulerIdentity, case.attempt.scheduler)
    fake_slurm_runner.set_task_state(
        scheduler,
        queue_state="RUNNING",
        accounting_state="FAILED",
        exit_code="1:0",
    )

    status = SlurmStateReconciler(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(fake_slurm_runner),
    ).refresh(observed_at=case.created_at + timedelta(minutes=3))

    attempt = status.shards[0].attempts[0]
    assert attempt.scheduler is not None and attempt.scheduler.state is SchedulerState.FAILED
    assert attempt.effective_state is EffectiveAttemptState.FAILED
    assert attempt.generation_state is GenerationState.FAILED
    assert status.effective_state is EffectiveRunState.FAILED


def test_refresh_rejects_winner_that_conflicts_with_terminal_scheduler_evidence(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    SlurmCommandClient(fake_slurm_runner).submit("run.sbatch")
    completed, winner = _publish_winner_state(case)
    scheduler = cast(SchedulerIdentity, completed.scheduler)
    fake_slurm_runner.set_task_state(
        scheduler,
        queue_state="RUNNING",
        accounting_state="NODE_FAIL",
        exit_code="1:0",
    )

    with pytest.raises(StateCorruptionError, match="winner conflicts"):
        SlurmStateReconciler(
            case.workspace,
            case.plan.run_id,
            SlurmCommandClient(fake_slurm_runner),
        ).refresh(observed_at=winner.published_at + timedelta(minutes=1))


def test_refresh_reports_a_validated_winner_as_succeeded(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    SlurmCommandClient(fake_slurm_runner).submit("run.sbatch")
    completed, winner = _publish_winner_state(case)
    scheduler = cast(SchedulerIdentity, completed.scheduler)
    fake_slurm_runner.set_task_state(
        scheduler,
        queue_state=None,
        accounting_state="COMPLETED",
        exit_code="0:0",
    )

    status = SlurmStateReconciler(
        case.workspace,
        case.plan.run_id,
        SlurmCommandClient(fake_slurm_runner),
    ).refresh(observed_at=winner.published_at + timedelta(minutes=1))

    attempt = status.shards[0].attempts[0]
    assert attempt.client_result is not None
    assert attempt.candidate_output is not None
    assert attempt.effective_state is EffectiveAttemptState.SUCCEEDED
    assert attempt.generation_state is GenerationState.WON
    assert status.shards[0].winner == winner
    assert status.effective_state is EffectiveRunState.SUCCEEDED


def test_refresh_rejects_a_concurrent_attempt_change_instead_of_guessing_status(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    scheduler = cast(SchedulerIdentity, case.attempt.scheduler)
    observed_at = case.created_at + timedelta(minutes=4)

    class MutatingClient:
        def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
            del selectors
            return (SlurmQueueEntry(job_identity=scheduler, state=SchedulerState.RUNNING),)

        def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
            del selectors
            case.writer.update_attempt(
                _copy_attempt(case.attempt, state=AttemptLifecycleState.RUNNING, updated_at=observed_at)
            )
            return ()

    with pytest.raises(StateConflictError, match="changed during reconciliation"):
        SlurmStateReconciler(case.workspace, case.plan.run_id, MutatingClient()).refresh(observed_at=observed_at)


def test_terminal_observation_remains_authoritative_during_later_accounting_gap() -> None:
    task = SchedulerIdentity(array_job_id=4101, array_task_id=0)
    first_time = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    first = SchedulerObservationCollector(
        _StaticSchedulerClient((), (_accounting(task, SchedulerState.COMPLETED),))
    ).collect((task,), observed_at=first_time)[0]
    later = SchedulerObservationCollector(
        _StaticSchedulerClient((SlurmQueueEntry(job_identity=task, state=SchedulerState.RUNNING),), ())
    ).collect((task,), observed_at=first_time + timedelta(minutes=1), previous={task: first})[0]

    assert later.state is SchedulerState.COMPLETED


def test_collector_rejects_conflicting_terminal_accounting_evidence() -> None:
    task = SchedulerIdentity(array_job_id=4101, array_task_id=0)
    first_time = datetime(2026, 9, 2, 12, tzinfo=timezone.utc)
    previous = SchedulerObservationCollector(
        _StaticSchedulerClient((), (_accounting(task, SchedulerState.COMPLETED),))
    ).collect((task,), observed_at=first_time)[0]

    with pytest.raises(SlurmStateError, match="violates persisted chronology"):
        SchedulerObservationCollector(_StaticSchedulerClient((), (_accounting(task, SchedulerState.FAILED),))).collect(
            (task,), observed_at=first_time + timedelta(minutes=1), previous={task: previous}
        )


def test_refresh_normalizes_scheduler_query_failures(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)

    class FailingClient:
        def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
            del selectors
            raise RuntimeError("scheduler unavailable")

        def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
            del selectors
            return ()

    with pytest.raises(SlurmStateError, match="cannot query normalized scheduler observations"):
        SlurmStateReconciler(case.workspace, case.plan.run_id, FailingClient()).refresh(
            observed_at=case.created_at + timedelta(minutes=3)
        )


def test_refresh_keeps_an_unsubmitted_attempt_pending_without_querying_slurm(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan, submitted=False)

    class UnexpectedClient:
        def query_queue(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmQueueEntry, ...]:
            raise AssertionError(f"unexpected queue query for {selectors!r}")

        def query_accounting(self, selectors: Sequence[SchedulerJobIdentity]) -> tuple[SlurmAccountingEntry, ...]:
            raise AssertionError(f"unexpected accounting query for {selectors!r}")

    status = SlurmStateReconciler(case.workspace, case.plan.run_id, UnexpectedClient()).refresh(
        observed_at=case.created_at + timedelta(minutes=3)
    )

    attempt = status.shards[0].attempts[0]
    assert attempt.scheduler is None
    assert attempt.effective_state is EffectiveAttemptState.PENDING
    assert attempt.generation_state is GenerationState.NOT_STARTED
    assert status.effective_state is EffectiveRunState.PENDING


def test_fresh_process_refresh_rejects_mismatched_persisted_scheduler_evidence(
    tmp_path: Path,
    authored_run_single: DataDesignerSlurmConfig,
    single_node_plan: ResolvedSlurmRunPlan,
    fake_slurm_runner: FakeSlurmRunner,
) -> None:
    case = _initialized_case(tmp_path, authored_run_single, single_node_plan)
    SlurmCommandClient(fake_slurm_runner).submit("run.sbatch")
    scheduler = cast(SchedulerIdentity, case.attempt.scheduler)
    fake_slurm_runner.set_task_state(scheduler, queue_state="RUNNING", accounting_state=None)
    observed_at = case.created_at + timedelta(minutes=3)
    SlurmStateReconciler(case.workspace, case.plan.run_id, SlurmCommandClient(fake_slurm_runner)).refresh(
        observed_at=observed_at
    )
    scheduler_path = case.writer.run_root / "shards/shard-00000/attempts/attempt-0001/scheduler.json"
    mismatched = SchedulerObservation(
        schema_version=1,
        scheduler=SchedulerIdentity(array_job_id=9999, array_task_id=0),
        observed_at=observed_at,
        state=SchedulerState.RUNNING,
    )
    scheduler_path.write_text(mismatched.serialize_json())

    with pytest.raises(StateCorruptionError, match="mismatched scheduler evidence"):
        SlurmStateReconciler(
            case.workspace,
            case.plan.run_id,
            SlurmCommandClient(fake_slurm_runner),
        ).refresh(observed_at=observed_at + timedelta(minutes=1))


def _accounting(identity: SchedulerJobIdentity, state: SchedulerState) -> SlurmAccountingEntry:
    return SlurmAccountingEntry(
        job_identity=identity,
        state=state,
        process_exit_code=SlurmProcessExitCode(exit_status=0, termination_signal=0),
    )


def _initialized_case(
    tmp_path: Path,
    authored_config: DataDesignerSlurmConfig,
    plan: ResolvedSlurmRunPlan,
    *,
    submitted: bool = True,
) -> _ReconciliationCase:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
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
        shard_count=1,
    )
    planned_shard = relocated_plan.shards[0]
    shard = ShardManifest(
        schema_version=1,
        run_id=relocated_plan.run_id,
        shard_id=planned_shard.shard_id,
        shard_index=planned_shard.shard_index,
        record_range=planned_shard.record_range,
        input_partition=planned_shard.input_partition,
        resume_workspace=planned_shard.resume_workspace,
        created_at=created_at,
    )
    writer = SlurmStateWriter(workspace, relocated_plan.run_id)
    writer.initialize_run(authored_config, relocated_plan, run, (shard,))
    attempt = AttemptManifest(
        schema_version=1,
        run_id=relocated_plan.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=run.resolved_plan,
        state=AttemptLifecycleState.SUBMITTED if submitted else AttemptLifecycleState.CREATED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=0) if submitted else None,
        created_at=created_at + timedelta(minutes=1),
        updated_at=created_at + timedelta(minutes=2),
    )
    writer.create_attempt(attempt)
    return _ReconciliationCase(workspace, relocated_plan, run, shard, attempt, writer, created_at)


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


def _publish_winner_state(case: _ReconciliationCase) -> tuple[AttemptManifest, ShardWinner]:
    running = _copy_attempt(
        case.attempt,
        state=AttemptLifecycleState.RUNNING,
        updated_at=case.created_at + timedelta(minutes=3),
    )
    case.writer.update_attempt(running)
    candidate_path = case.writer.run_root / "shards/shard-00000/attempts/attempt-0001/output-manifest.json"
    dataset_path = candidate_path.parent / "dataset"
    requested = case.plan.shards[0].requested_records
    candidate = CandidateOutputManifest(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id=running.shard_id,
        attempt_id=running.attempt_id,
        attempt_ordinal=running.attempt_ordinal,
        created_at=case.created_at + timedelta(minutes=4),
        dataset_path=dataset_path.as_posix(),
        requested_records=requested,
        actual_records=requested,
        outcome=CandidateOutcome.COMPLETE,
        files=(
            CandidateOutputFile(
                relative_path="part-00000.parquet",
                sha256="a" * 64,
                byte_size=1,
                record_count=requested,
            ),
        ),
        dataset_schema_digest="b" * 64,
        provenance_digest=case.plan.compute_sha256(),
    )
    candidate_reference = ArtifactReference(path=candidate_path.as_posix(), sha256=candidate.compute_sha256())
    result = ClientResult(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id=running.shard_id,
        attempt_id=running.attempt_id,
        completed_at=case.created_at + timedelta(minutes=5),
        requested_records=requested,
        actual_records=requested,
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
        updated_at=case.created_at + timedelta(minutes=6),
    )
    case.writer.update_attempt(completed)
    winner = ShardWinner(
        schema_version=1,
        run_id=case.plan.run_id,
        shard_id=completed.shard_id,
        attempt_id=completed.attempt_id,
        attempt_ordinal=completed.attempt_ordinal,
        candidate_manifest=candidate_reference,
        published_at=case.created_at + timedelta(minutes=7),
    )
    winner_path = case.writer.run_root / "shards/shard-00000/winner.json"
    winner_path.write_text(winner.serialize_json())
    winner_path.chmod(0o600)
    return completed, winner


def _copy_attempt(attempt: AttemptManifest, **updates: object) -> AttemptManifest:
    payload = attempt.model_dump(mode="json")
    payload.update(updates)
    return AttemptManifest.model_validate_json(json.dumps(payload, default=_json_value))


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported test value: {type(value).__name__}")
