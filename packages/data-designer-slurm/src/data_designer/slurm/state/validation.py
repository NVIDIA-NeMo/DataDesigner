# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.slurm.state.execution import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
    RunManifest,
    ShardManifest,
)
from data_designer.slurm.state.outputs import (
    CandidateOutputManifest,
    CollectionPlan,
    ShardWinner,
)
from data_designer.slurm.state.scheduler import SchedulerObservation, SchedulerState

_ATTEMPT_STATE_ORDER = {
    AttemptLifecycleState.CREATED: 0,
    AttemptLifecycleState.SUBMITTED: 1,
    AttemptLifecycleState.PENDING: 2,
    AttemptLifecycleState.RUNNING: 3,
}
_TERMINAL_ATTEMPT_STATES = frozenset(
    {
        AttemptLifecycleState.SUCCEEDED,
        AttemptLifecycleState.FAILED,
    }
)


class StateContractError(ValueError):
    """Raised when related state records violate a cross-record contract."""


def validate_shard_manifest(run: RunManifest, shard: ShardManifest) -> ShardManifest:
    """Validate a shard against the immutable run identity and bounds."""
    _require(shard.run_id == run.run_id, "shard run_id does not match run manifest")
    _require(shard.shard_index < run.shard_count, "shard_index is outside the run shard count")
    _require(shard.created_at >= run.created_at, "shard creation cannot precede run creation")
    return shard


def validate_shard_set(
    run: RunManifest,
    shards: tuple[ShardManifest, ...],
) -> tuple[ShardManifest, ...]:
    """Validate ordered run state shards without duplicating resolved-plan coverage."""
    _require(len(shards) == run.shard_count, "shard set must include exactly the run shard count")
    for shard in shards:
        validate_shard_manifest(run, shard)

    shard_ids = tuple(shard.shard_id for shard in shards)
    resume_workspace_ids = tuple(shard.resume_workspace_id for shard in shards)
    shard_indices = tuple(shard.shard_index for shard in shards)
    _require(len(set(shard_ids)) == len(shards), "shard IDs must be unique")
    _require(len(set(resume_workspace_ids)) == len(shards), "resume workspace IDs must be unique")
    _require(
        shard_indices == tuple(range(run.shard_count)),
        "shards must be ordered by a complete zero-based shard index",
    )
    for previous, current in zip(shards, shards[1:]):
        _require(
            current.record_range.start_index >= previous.record_range.end_index_exclusive,
            "shard record ranges must not overlap",
        )
    return shards


def validate_attempt_manifest(
    run: RunManifest,
    shard: ShardManifest,
    attempt: AttemptManifest,
) -> AttemptManifest:
    """Validate an attempt against its run, shard, and resolved plan."""
    validate_shard_manifest(run, shard)
    _require(attempt.run_id == run.run_id, "attempt run_id does not match run manifest")
    _require(attempt.shard_id == shard.shard_id, "attempt shard_id does not match shard manifest")
    _require(
        attempt.resolved_plan == run.resolved_plan,
        "attempt resolved plan does not match run manifest",
    )
    _require(attempt.created_at >= shard.created_at, "attempt creation cannot precede shard creation")
    return attempt


def validate_attempt_set(
    run: RunManifest,
    shards: tuple[ShardManifest, ...],
    attempts: tuple[AttemptManifest, ...],
) -> tuple[AttemptManifest, ...]:
    """Validate attempt identities and scheduler ownership across a run."""
    validate_shard_set(run, shards)
    shard_by_id = {shard.shard_id: shard for shard in shards}

    for attempt in attempts:
        shard = shard_by_id.get(attempt.shard_id)
        _require(shard is not None, f"attempt references unknown shard {attempt.shard_id!r}")
        validate_attempt_manifest(run, shard, attempt)

    attempt_ids = tuple((attempt.shard_id, attempt.attempt_id) for attempt in attempts)
    shard_ordinals = tuple((attempt.shard_id, attempt.attempt_ordinal) for attempt in attempts)
    scheduler_identities = tuple(attempt.scheduler for attempt in attempts if attempt.scheduler is not None)
    _require(len(set(attempt_ids)) == len(attempts), "attempt IDs must be unique within each shard")
    _require(
        len(set(shard_ordinals)) == len(attempts),
        "attempt ordinals must be unique within each shard",
    )
    _require(
        len(set(scheduler_identities)) == len(scheduler_identities),
        "scheduler identities must be unique across attempts",
    )
    return attempts


def validate_attempt_transition(
    previous: AttemptManifest,
    current: AttemptManifest,
) -> AttemptManifest:
    """Validate immutable identity and monotonic lifecycle updates for an attempt."""
    for field_name in (
        "run_id",
        "shard_id",
        "attempt_id",
        "attempt_ordinal",
        "resolved_plan",
        "created_at",
    ):
        _require(
            getattr(previous, field_name) == getattr(current, field_name),
            f"attempt {field_name} cannot change",
        )
    _require(current.updated_at >= previous.updated_at, "attempt updated_at cannot move backward")

    if previous.state in _TERMINAL_ATTEMPT_STATES:
        _require(current == previous, "terminal attempt manifests are immutable")
        return current

    if current.state not in _TERMINAL_ATTEMPT_STATES:
        _require(
            _ATTEMPT_STATE_ORDER[current.state] >= _ATTEMPT_STATE_ORDER[previous.state],
            f"attempt state cannot move from {previous.state.value} to {current.state.value}",
        )
    if previous.scheduler is not None:
        _require(current.scheduler == previous.scheduler, "attempt scheduler identity cannot change")
    if previous.candidate_output is not None:
        _require(current.candidate_output == previous.candidate_output, "attempt candidate output cannot change")
    return current


def validate_scheduler_observation_transition(
    previous: SchedulerObservation,
    current: SchedulerObservation,
) -> SchedulerObservation:
    """Validate scheduler identity, chronology, and a fixed accounting-lag deadline."""
    _require(current.scheduler == previous.scheduler, "scheduler identity cannot change between observations")
    _require(current.observed_at >= previous.observed_at, "scheduler observed_at cannot move backward")

    if previous.state is SchedulerState.ACCOUNTING_LAG:
        deadline = previous.reconciliation_deadline
        _require(deadline is not None, "accounting lag has no reconciliation deadline")
        if current.state is SchedulerState.ACCOUNTING_LAG:
            _require(
                current.reconciliation_deadline == deadline,
                "accounting-lag reconciliation deadline cannot change",
            )
        elif current.state is SchedulerState.UNKNOWN:
            _require(
                current.observed_at > deadline,
                "accounting lag cannot become unknown before its reconciliation deadline expires",
            )
    return current


def validate_shard_winner(
    run: RunManifest,
    shard: ShardManifest,
    attempt: AttemptManifest,
    candidate: CandidateOutputManifest,
    winner: ShardWinner,
    *,
    existing_winner: ShardWinner | None = None,
) -> ShardWinner:
    """Validate first-writer winner publication without performing persistence."""
    validate_attempt_manifest(run, shard, attempt)
    _require(existing_winner is None, "a shard winner is immutable once published")

    for record_name, record_run_id in (
        ("candidate", candidate.run_id),
        ("winner", winner.run_id),
    ):
        _require(record_run_id == run.run_id, f"{record_name} run_id does not match run manifest")
    for record_name, record_shard_id in (
        ("candidate", candidate.shard_id),
        ("winner", winner.shard_id),
    ):
        _require(record_shard_id == shard.shard_id, f"{record_name} shard_id does not match shard manifest")

    _require(candidate.attempt_id == attempt.attempt_id, "candidate attempt_id does not match attempt")
    _require(winner.attempt_id == attempt.attempt_id, "winner attempt_id does not match attempt")
    _require(
        candidate.attempt_ordinal == attempt.attempt_ordinal == winner.attempt_ordinal,
        "candidate and winner attempt ordinals must match the attempt",
    )
    _require(attempt.state is AttemptLifecycleState.SUCCEEDED, "only successful attempts may win")
    _require(
        attempt.terminal_classification is AttemptTerminalClassification.SUCCEEDED,
        "only successfully classified attempts may win",
    )
    _require(candidate.winner_eligible, "candidate output does not satisfy winner policy")
    _require(
        candidate.requested_records == shard.record_range.record_count,
        "candidate requested record count does not match shard record range",
    )
    _require(candidate.created_at >= attempt.created_at, "candidate creation cannot precede attempt creation")
    _require(attempt.updated_at >= candidate.created_at, "attempt completion cannot precede candidate creation")
    _require(winner.published_at >= attempt.updated_at, "winner publication cannot precede attempt completion")
    _require(
        winner.candidate_manifest.sha256 == candidate.compute_sha256(),
        "winner candidate digest does not match candidate manifest",
    )
    _require(
        attempt.candidate_output == winner.candidate_manifest,
        "attempt candidate output reference does not match winner",
    )
    return winner


def validate_collection_plan(
    run: RunManifest,
    plan: CollectionPlan,
    shards: tuple[ShardManifest, ...],
    winners: tuple[ShardWinner, ...],
) -> CollectionPlan:
    """Validate collection against the exact run shard set and winner digests."""
    validate_shard_set(run, shards)
    _require(plan.run_id == run.run_id, "collection run_id does not match run manifest")
    _require(plan.resolved_plan == run.resolved_plan, "collection resolved plan does not match run manifest")
    _require(plan.created_at >= run.created_at, "collection creation cannot precede run creation")

    expected_shard_ids = tuple(shard.shard_id for shard in shards)
    planned_shard_ids = tuple(shard.shard_id for shard in plan.planned_shards)
    _require(
        planned_shard_ids == expected_shard_ids,
        "collection planned shards must exactly match the ordered run shard set",
    )
    _require(len(winners) == len(shards), "winner set must include exactly the run shard count")

    winner_by_shard = {winner.shard_id: winner for winner in winners}
    _require(len(winner_by_shard) == len(winners), "winner shard IDs must be unique")
    _require(set(winner_by_shard) == set(expected_shard_ids), "winner shard IDs must exactly match the run shard set")
    for planned_shard in plan.planned_shards:
        winner = winner_by_shard[planned_shard.shard_id]
        _require(winner.run_id == run.run_id, "winner run_id does not match run manifest")
        _require(plan.created_at >= winner.published_at, "collection creation cannot precede winner publication")
        _require(
            planned_shard.winner_manifest.sha256 == winner.compute_sha256(),
            f"winner digest mismatch for shard {planned_shard.shard_id!r}",
        )
    return plan


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateContractError(message)
