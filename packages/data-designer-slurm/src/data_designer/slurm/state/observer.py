# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process reconciliation of one persisted Slurm run."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.contracts import Identifier, ShardId, validate_absolute_path
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.base import SchedulerIdentity, SchedulerJobIdentity
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.finalization import WinnerFinalizer
from data_designer.slurm.state.observation import SchedulerObservationClient, SchedulerObservationCollector
from data_designer.slurm.state.outputs import ShardWinner
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.reconciliation import reconcile_attempt_observation
from data_designer.slurm.state.scheduler import EffectiveAttemptState, SchedulerObservation
from data_designer.slurm.state.status import (
    AttemptStatus,
    RunStatus,
    ShardStatus,
    derive_generation_state,
    derive_run_state,
    derive_shard_state,
)
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import StateContractError, validate_scheduler_observation_transition

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)


@dataclass(frozen=True, slots=True)
class _ShardSnapshot:
    run: RunManifest
    plan: ResolvedSlurmRunPlan
    shard: ShardManifest
    attempts: tuple[AttemptManifest, ...]


@dataclass(frozen=True, slots=True)
class _ObservationBatch:
    previous: dict[SchedulerIdentity, SchedulerObservation | None]
    current: dict[SchedulerJobIdentity, SchedulerObservation]
    observed_at: datetime


class SlurmStateReconciler:
    """Refresh persisted run status from normalized scheduler observations.

    Each refresh reconstructs state from the compute-visible workspace. No
    controller memory participates in status, wait, or benchmark refreshes.

    Args:
        workspace_root: Selected compute-visible workspace root.
        run_id: Stable application-owned run identity.
        scheduler: Client returning normalized active and accounting records.
    """

    def __init__(
        self,
        workspace_root: str | Path,
        run_id: Identifier,
        scheduler: SchedulerObservationClient,
    ) -> None:
        normalized_root, normalized_run_id = _validate_location(workspace_root, run_id)
        self._storage = StateStorage(normalized_root, normalized_run_id)
        self._reader = StateReader(self._storage, normalized_run_id)
        self._finalizer = WinnerFinalizer(self._storage, self._reader)
        self._collector = SchedulerObservationCollector(scheduler)
        self._run_id = normalized_run_id

    @property
    def run_root(self) -> Path:
        """Return the workspace-derived root for this run."""
        return self._storage.run_root

    def refresh(self, *, observed_at: datetime | None = None) -> RunStatus:
        """Persist current scheduler evidence and return complete run status.

        Raises:
            SlurmStateError: If scheduler evidence cannot be queried or state
                cannot be reconstructed safely.
        """
        timestamp = datetime.now(timezone.utc) if observed_at is None else observed_at
        _validate_observed_at(timestamp)
        run, plan, shards = self._reader.load_context()
        if timestamp < run.created_at:
            raise SlurmStateError("observation timestamp cannot precede run creation")
        attempts_by_shard = self._reader.load_validated_attempts(run, plan, shards)
        previous = self._load_previous_observations(attempts_by_shard)
        selectors = tuple(previous.keys())
        current = self._collector.collect(selectors, observed_at=timestamp, previous=previous)
        batch = _ObservationBatch(
            previous=previous,
            current={observation.scheduler: observation for observation in current},
            observed_at=timestamp,
        )
        shard_statuses = tuple(
            self._refresh_shard(
                _ShardSnapshot(run, plan, shard, attempts_by_shard[shard.shard_id]),
                batch,
            )
            for shard in shards
        )
        return self._compose_run_status(run, timestamp, shard_statuses)

    def _compose_run_status(
        self,
        run: RunManifest,
        observed_at: datetime,
        shards: tuple[ShardStatus, ...],
    ) -> RunStatus:
        try:
            return RunStatus(
                run=run,
                observed_at=observed_at,
                shards=shards,
                effective_state=derive_run_state(shards),
            )
        except ValidationError as error:
            raise StateCorruptionError(f"cannot reconcile run {self._run_id!r}") from error

    def _load_previous_observations(
        self,
        attempts_by_shard: dict[ShardId, tuple[AttemptManifest, ...]],
    ) -> dict[SchedulerIdentity, SchedulerObservation | None]:
        previous: dict[SchedulerIdentity, SchedulerObservation | None] = {}
        for attempts in attempts_by_shard.values():
            for attempt in attempts:
                if attempt.scheduler is not None:
                    previous[attempt.scheduler] = self._reader.load_optional_scheduler_observation(attempt)
        return previous

    def _refresh_shard(
        self,
        expected: _ShardSnapshot,
        batch: _ObservationBatch,
    ) -> ShardStatus:
        try:
            with self._storage.acquire_shard_lock(expected.shard.shard_id):
                current_run, current_plan, current_shard = self._reader.load_shard_context(expected.shard.shard_id)
                attempts = self._reader.load_validated_shard_attempts(current_run, current_plan, current_shard)
                self._require_unchanged_context(
                    expected,
                    _ShardSnapshot(current_run, current_plan, current_shard, attempts),
                )
                winner = self._finalizer.load_optional_winner(
                    expected.run,
                    expected.plan,
                    expected.shard,
                    attempts,
                )
                statuses = tuple(
                    self._build_attempt_status(
                        expected,
                        batch,
                        attempt,
                        winner,
                    )
                    for attempt in attempts
                )
                self._validate_winner_scheduler_consistency(winner, statuses)
                return ShardStatus(
                    shard=expected.shard,
                    attempts=statuses,
                    winner=winner,
                    effective_state=derive_shard_state(statuses, winner),
                )
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (OSError, StateContractError, ValidationError) as error:
            raise StateCorruptionError(f"cannot reconcile shard {expected.shard.shard_id!r}") from error

    def _build_attempt_status(
        self,
        snapshot: _ShardSnapshot,
        batch: _ObservationBatch,
        attempt: AttemptManifest,
        winner: ShardWinner | None,
    ) -> AttemptStatus:
        readiness = self._reader.load_optional_readiness(snapshot.plan, attempt)
        result = self._reader.load_optional_attempt_result(snapshot.plan, snapshot.shard, attempt)
        scheduler = None
        if attempt.scheduler is not None:
            scheduler = batch.current[attempt.scheduler]
            self._persist_observation(attempt, batch.previous[attempt.scheduler], scheduler)
            effective_state = reconcile_attempt_observation(
                attempt,
                readiness,
                scheduler,
                current_time=batch.observed_at,
            )
        else:
            if attempt.state is not AttemptLifecycleState.CREATED:
                raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has no scheduler identity")
            effective_state = EffectiveAttemptState.PENDING
        if attempt.candidate_output is not None and result is None:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} is missing its generation results")
        client_result, candidate = (None, None) if result is None else result
        is_winner = winner is not None and winner.attempt_id == attempt.attempt_id
        generation_state = derive_generation_state(
            effective_state,
            has_candidate=candidate is not None,
            is_winner=is_winner,
        )
        return AttemptStatus(
            attempt=attempt,
            readiness=readiness,
            scheduler=scheduler,
            client_result=client_result,
            candidate_output=candidate,
            effective_state=effective_state,
            generation_state=generation_state,
            is_winner=is_winner,
        )

    def _persist_observation(
        self,
        attempt: AttemptManifest,
        expected_previous: SchedulerObservation | None,
        current: SchedulerObservation,
    ) -> None:
        persisted = self._reader.load_optional_scheduler_observation(attempt)
        if persisted != expected_previous:
            raise StateConflictError("scheduler evidence changed during reconciliation; refresh again")
        if persisted == current:
            self._storage.sync_attempt_directory(attempt.shard_id, attempt.attempt_id)
            return
        if persisted is None:
            self._storage.publish_scheduler_observation(attempt.shard_id, attempt.attempt_id, current)
            return
        validate_scheduler_observation_transition(persisted, current)
        self._storage.replace_scheduler_observation(attempt.shard_id, attempt.attempt_id, current)

    @staticmethod
    def _require_unchanged_context(
        expected: _ShardSnapshot,
        current: _ShardSnapshot,
    ) -> None:
        if current != expected:
            raise StateConflictError("persisted state changed during reconciliation; refresh again")

    @staticmethod
    def _validate_winner_scheduler_consistency(
        winner: ShardWinner | None,
        statuses: tuple[AttemptStatus, ...],
    ) -> None:
        if winner is None:
            return
        winning = next(status for status in statuses if status.attempt.attempt_id == winner.attempt_id)
        if winning.effective_state is not EffectiveAttemptState.SUCCEEDED:
            raise StateCorruptionError("persisted winner conflicts with terminal scheduler evidence")


def _validate_location(workspace_root: str | Path, run_id: Identifier) -> tuple[Path, Identifier]:
    try:
        normalized_root = validate_absolute_path(Path(workspace_root).as_posix())
        normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
    except (ValidationError, ValueError) as error:
        raise SlurmStateError("invalid persisted run location") from error
    return Path(normalized_root), normalized_run_id


def _validate_observed_at(observed_at: datetime) -> None:
    if not isinstance(observed_at, datetime) or observed_at.tzinfo is None or observed_at.utcoffset() is None:
        raise SlurmStateError("observation timestamp must be timezone-aware UTC")
    if observed_at.utcoffset().total_seconds() != 0:
        raise SlurmStateError("observation timestamp must be timezone-aware UTC")


__all__ = ["SlurmStateReconciler"]
