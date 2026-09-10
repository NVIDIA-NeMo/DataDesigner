# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process observation of ordinary benchmark child state."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from data_designer.slurm.benchmark.analysis import (
    BenchmarkMeasurements,
    BenchmarkObservationFailure,
    BenchmarkRunObservation,
)
from data_designer.slurm.benchmark.records import BenchmarkOutcome
from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    EffectiveRunState,
    ProbeOutcome,
    ReadinessState,
    RunManifest,
    RunStatus,
    SchedulerObservationClient,
    SchedulerState,
    SchedulerStateConflictError,
    ShardManifest,
    SlurmStateError,
    SlurmStateReconciler,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.finalization import WinnerFinalizer
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage

_MeasurementRecord = tuple[
    AttemptManifest,
    AttemptReadiness | None,
    tuple[ClientResult, CandidateOutputManifest] | None,
]


class PersistedBenchmarkRunObserver:
    """Reconstruct benchmark child facts from the ordinary run state tree."""

    def __init__(
        self,
        workspace_root: str | Path,
        scheduler: SchedulerObservationClient,
        clock: Callable[[], datetime],
    ) -> None:
        self._workspace_root = Path(workspace_root)
        self._scheduler = scheduler
        self._clock = clock

    def observe(self, run_id: str, *, refresh_state: bool) -> BenchmarkRunObservation:
        try:
            storage = StateStorage(self._workspace_root, run_id)
            reader = StateReader(storage, run_id)
            run, plan, shards = reader.load_context()
            authored = reader.load_authored_config(run)
            if refresh_state:
                status = SlurmStateReconciler(
                    self._workspace_root,
                    run_id,
                    self._scheduler,
                ).refresh(observed_at=self._clock())
                outcome = _status_outcome(status.effective_state)
                measurements = _measure_status(status) if outcome is BenchmarkOutcome.SUCCEEDED else None
            else:
                outcome, measurements = _load_persisted_outcome(storage, reader, run, plan, shards)
            return BenchmarkRunObservation(
                authored_config=authored,
                resolved_plan=plan,
                outcome=outcome,
                measurements=measurements,
            )
        except StateNotFoundError:
            raise BenchmarkObservationFailure(BenchmarkOutcome.MISSING) from None
        except SchedulerStateConflictError:
            raise BenchmarkObservationFailure(BenchmarkOutcome.SCHEDULER_INCONSISTENT) from None
        except (StateConflictError, StateCorruptionError):
            raise BenchmarkObservationFailure(BenchmarkOutcome.STALE) from None
        except (OSError, SlurmStateError):
            raise BenchmarkObservationFailure(BenchmarkOutcome.STALE) from None


def _load_persisted_outcome(
    storage: StateStorage,
    reader: StateReader,
    run: RunManifest,
    plan: ResolvedSlurmRunPlan,
    shards: tuple[ShardManifest, ...],
) -> tuple[BenchmarkOutcome, BenchmarkMeasurements | None]:
    attempts_by_shard = reader.load_validated_attempts(run, plan, shards)
    finalizer = WinnerFinalizer(storage, reader)
    winning_records: list[_MeasurementRecord] = []
    latest_states: list[AttemptLifecycleState] = []
    scheduler_states: list[SchedulerState] = []
    for shard in shards:
        attempts = attempts_by_shard[shard.shard_id]
        if not attempts:
            return BenchmarkOutcome.INCOMPLETE, None
        winner = finalizer.load_optional_winner(run, plan, shard, attempts)
        if winner is None:
            latest = attempts[-1]
            latest_states.append(latest.state)
            observation = reader.load_optional_scheduler_observation(latest)
            if observation is not None:
                scheduler_states.append(observation.state)
            continue
        attempt = next(item for item in attempts if item.attempt_id == winner.attempt_id)
        observation = reader.load_optional_scheduler_observation(attempt)
        if observation is not None and observation.state in {
            SchedulerState.FAILED,
            SchedulerState.CANCELLED,
            SchedulerState.TIMED_OUT,
            SchedulerState.NODE_FAILED,
            SchedulerState.OUT_OF_MEMORY,
        }:
            raise SchedulerStateConflictError("persisted winner conflicts with scheduler evidence")
        result = reader.load_optional_attempt_result(plan, shard, attempt)
        readiness = reader.load_optional_readiness(plan, attempt)
        winning_records.append((attempt, readiness, result))

    if len(winning_records) == len(shards):
        return BenchmarkOutcome.SUCCEEDED, _measure_records(tuple(winning_records))
    if SchedulerState.ACCOUNTING_LAG in scheduler_states:
        return BenchmarkOutcome.ACCOUNTING_LAG, None
    if any(
        state in {SchedulerState.UNKNOWN, SchedulerState.PREEMPTED, SchedulerState.REQUEUED}
        for state in scheduler_states
    ):
        return BenchmarkOutcome.STALE, None
    if any(
        state
        in {
            AttemptLifecycleState.CREATED,
            AttemptLifecycleState.SUBMITTED,
            AttemptLifecycleState.PENDING,
            AttemptLifecycleState.RUNNING,
        }
        for state in latest_states
    ):
        return BenchmarkOutcome.PENDING, None
    if all(state is AttemptLifecycleState.FAILED for state in latest_states):
        return BenchmarkOutcome.FAILED, None
    return BenchmarkOutcome.INCOMPLETE, None


def _status_outcome(state: EffectiveRunState) -> BenchmarkOutcome:
    if state in {EffectiveRunState.PENDING, EffectiveRunState.RUNNING}:
        return BenchmarkOutcome.PENDING
    if state is EffectiveRunState.ACCOUNTING_LAG:
        return BenchmarkOutcome.ACCOUNTING_LAG
    if state is EffectiveRunState.SUCCEEDED:
        return BenchmarkOutcome.SUCCEEDED
    if state is EffectiveRunState.FAILED:
        return BenchmarkOutcome.FAILED
    return BenchmarkOutcome.STALE


def _measure_status(status: RunStatus) -> BenchmarkMeasurements | None:
    records: list[_MeasurementRecord] = []
    for shard in status.shards:
        winner = next((attempt for attempt in shard.attempts if attempt.is_winner), None)
        if winner is None:
            return None
        result = (
            None
            if winner.client_result is None or winner.candidate_output is None
            else (winner.client_result, winner.candidate_output)
        )
        records.append((winner.attempt, winner.readiness, result))
    return _measure_records(tuple(records))


def _measure_records(records: tuple[_MeasurementRecord, ...]) -> BenchmarkMeasurements | None:
    attempts: list[AttemptManifest] = []
    started_times: list[datetime] = []
    ready_times: list[datetime] = []
    completed_times: list[datetime] = []
    actual_records = 0
    for attempt, readiness, result in records:
        if result is None:
            return None
        client_result, candidate = result
        if (
            client_result.outcome is not ClientOutcome.COMPLETE
            or candidate.actual_records != client_result.actual_records
        ):
            return None
        if readiness is None or readiness.state not in {ReadinessState.READY, ReadinessState.STOPPED}:
            return None
        if readiness.started_at is None:
            return None
        probes = tuple(deployment.last_probe for deployment in readiness.deployments)
        if any(probe is None or probe.outcome is not ProbeOutcome.SUCCESS for probe in probes):
            return None
        attempts.append(attempt)
        started_times.append(readiness.started_at)
        ready_times.append(max(probe.observed_at for probe in probes if probe is not None))
        completed_times.append(client_result.completed_at)
        actual_records += client_result.actual_records or 0

    started_at = min(started_times)
    ready_at = max(ready_times)
    completed_at = max(completed_times)
    stopped_at = max(attempt.updated_at for attempt in attempts)
    if ready_at < started_at or completed_at <= ready_at or stopped_at <= started_at:
        return None
    return BenchmarkMeasurements(
        actual_records=actual_records,
        boot_seconds=(ready_at - started_at).total_seconds(),
        generation_seconds=(completed_at - ready_at).total_seconds(),
        wall_seconds=(stopped_at - started_at).total_seconds(),
    )


__all__ = ["PersistedBenchmarkRunObserver"]
