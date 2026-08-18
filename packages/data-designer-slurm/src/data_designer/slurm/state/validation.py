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
    """Validate the exact, ordered set of shards belonging to a run."""
    _require(len(shards) == run.shard_count, "shard set must include exactly the run shard count")
    for shard in shards:
        validate_shard_manifest(run, shard)

    shard_ids = tuple(shard.shard_id for shard in shards)
    shard_indices = tuple(shard.shard_index for shard in shards)
    _require(len(set(shard_ids)) == len(shards), "shard IDs must be unique")
    _require(
        shard_indices == tuple(range(run.shard_count)),
        "shards must be ordered by a complete zero-based shard index",
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
