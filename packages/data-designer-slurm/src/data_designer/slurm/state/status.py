# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fresh-process status values derived from persisted and scheduler evidence."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import Field, field_validator, model_validator

from data_designer.slurm.client import ClientResult
from data_designer.slurm.state.base import StateValue, validate_utc_timestamp
from data_designer.slurm.state.execution import AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.outputs import CandidateOutputManifest, ShardWinner
from data_designer.slurm.state.readiness import AttemptReadiness
from data_designer.slurm.state.scheduler import EffectiveAttemptState, SchedulerObservation


class GenerationState(str, Enum):
    """Effective progress of one attempt's dataset generation."""

    NOT_STARTED = "not_started"
    ACTIVE = "active"
    CANDIDATE_READY = "candidate_ready"
    WON = "won"
    FAILED = "failed"
    UNKNOWN = "unknown"


class EffectiveRunState(str, Enum):
    """Aggregated state of all planned shards in one run."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    ACCOUNTING_LAG = "accounting_lag"
    UNKNOWN = "unknown"


class AttemptStatus(StateValue):
    """Validated persisted and observed evidence for one attempt."""

    attempt: AttemptManifest
    readiness: AttemptReadiness | None
    scheduler: SchedulerObservation | None
    client_result: ClientResult | None
    candidate_output: CandidateOutputManifest | None
    effective_state: EffectiveAttemptState
    generation_state: GenerationState
    is_winner: bool = False

    @model_validator(mode="after")
    def validate_evidence(self) -> AttemptStatus:
        if self.attempt.scheduler is None:
            if self.scheduler is not None:
                raise ValueError("created attempts cannot have scheduler evidence")
        elif self.scheduler is None or self.scheduler.scheduler != self.attempt.scheduler:
            raise ValueError("attempt status requires matching scheduler evidence")
        if self.readiness is not None and (
            self.readiness.run_id,
            self.readiness.shard_id,
            self.readiness.attempt_id,
        ) != (self.attempt.run_id, self.attempt.shard_id, self.attempt.attempt_id):
            raise ValueError("attempt status readiness identity does not match")
        if (self.client_result is None) != (self.candidate_output is None):
            raise ValueError("attempt status requires a complete generation-result pair")
        if self.client_result is not None and self.candidate_output is not None:
            expected = (self.attempt.run_id, self.attempt.shard_id, self.attempt.attempt_id)
            if (
                self.client_result.run_id,
                self.client_result.shard_id,
                self.client_result.attempt_id,
            ) != expected:
                raise ValueError("client result identity does not match the attempt")
            if (
                self.candidate_output.run_id,
                self.candidate_output.shard_id,
                self.candidate_output.attempt_id,
            ) != expected:
                raise ValueError("candidate output identity does not match the attempt")
        expected_generation = derive_generation_state(
            self.effective_state,
            has_candidate=self.candidate_output is not None,
            is_winner=self.is_winner,
        )
        if self.generation_state is not expected_generation:
            raise ValueError("generation state does not match the attempt evidence")
        return self


class ShardStatus(StateValue):
    """Effective state and attempt history for one planned shard."""

    shard: ShardManifest
    attempts: tuple[AttemptStatus, ...]
    winner: ShardWinner | None
    effective_state: EffectiveAttemptState

    @model_validator(mode="after")
    def validate_status(self) -> ShardStatus:
        if any(status.attempt.shard_id != self.shard.shard_id for status in self.attempts):
            raise ValueError("shard status contains an attempt for another shard")
        ordinals = tuple(status.attempt.attempt_ordinal for status in self.attempts)
        if ordinals != tuple(range(1, len(self.attempts) + 1)):
            raise ValueError("shard status attempts must be in complete ordinal order")
        winning_attempts = tuple(status for status in self.attempts if status.is_winner)
        if self.winner is None:
            if winning_attempts:
                raise ValueError("shard status marks a winner without a winner record")
        elif (
            self.winner.shard_id != self.shard.shard_id
            or len(winning_attempts) != 1
            or winning_attempts[0].attempt.attempt_id != self.winner.attempt_id
        ):
            raise ValueError("shard winner does not match its observed attempt")
        if self.effective_state is not derive_shard_state(self.attempts, self.winner):
            raise ValueError("effective shard state does not match its evidence")
        return self


class RunStatus(StateValue):
    """Fresh-process status for every planned shard in one run."""

    run: RunManifest
    observed_at: datetime
    shards: tuple[ShardStatus, ...] = Field(min_length=1)
    effective_state: EffectiveRunState

    _observed_at_is_utc = field_validator("observed_at")(validate_utc_timestamp)

    @model_validator(mode="after")
    def validate_status(self) -> RunStatus:
        if self.observed_at < self.run.created_at:
            raise ValueError("run observation cannot precede run creation")
        if len(self.shards) != self.run.shard_count:
            raise ValueError("run status must include every planned shard")
        if tuple(status.shard.shard_index for status in self.shards) != tuple(range(self.run.shard_count)):
            raise ValueError("run status shards must be in planned order")
        if any(status.shard.run_id != self.run.run_id for status in self.shards):
            raise ValueError("run status contains a shard for another run")
        if self.effective_state is not derive_run_state(self.shards):
            raise ValueError("effective run state does not match its shard evidence")
        return self


def derive_generation_state(
    effective_state: EffectiveAttemptState,
    *,
    has_candidate: bool,
    is_winner: bool,
) -> GenerationState:
    """Derive generation progress without treating readiness as success."""
    if effective_state is EffectiveAttemptState.FAILED:
        return GenerationState.FAILED
    if effective_state is EffectiveAttemptState.UNKNOWN:
        return GenerationState.UNKNOWN
    if is_winner:
        return GenerationState.WON
    if has_candidate:
        return GenerationState.CANDIDATE_READY
    if effective_state is EffectiveAttemptState.RUNNING:
        return GenerationState.ACTIVE
    return GenerationState.NOT_STARTED


def derive_shard_state(
    attempts: tuple[AttemptStatus, ...],
    winner: ShardWinner | None,
) -> EffectiveAttemptState:
    """Derive one shard's effective state from its newest attempt and winner."""
    if winner is not None:
        winning = next((status for status in attempts if status.attempt.attempt_id == winner.attempt_id), None)
        if winning is not None and winning.effective_state is EffectiveAttemptState.SUCCEEDED:
            return EffectiveAttemptState.SUCCEEDED
    if not attempts:
        return EffectiveAttemptState.PENDING
    return attempts[-1].effective_state


def derive_run_state(shards: tuple[ShardStatus, ...]) -> EffectiveRunState:
    """Aggregate shard states without declaring a partially active run terminal."""
    states = tuple(shard.effective_state for shard in shards)
    if all(state is EffectiveAttemptState.SUCCEEDED for state in states):
        return EffectiveRunState.SUCCEEDED
    if any(state is EffectiveAttemptState.RUNNING for state in states):
        return EffectiveRunState.RUNNING
    if any(state is EffectiveAttemptState.ACCOUNTING_LAG for state in states):
        return EffectiveRunState.ACCOUNTING_LAG
    if any(state is EffectiveAttemptState.PENDING for state in states):
        return EffectiveRunState.PENDING
    if any(state is EffectiveAttemptState.UNKNOWN for state in states):
        return EffectiveRunState.UNKNOWN
    return EffectiveRunState.FAILED


__all__ = [
    "AttemptStatus",
    "EffectiveRunState",
    "GenerationState",
    "RunStatus",
    "ShardStatus",
    "derive_generation_state",
    "derive_run_state",
    "derive_shard_state",
]
