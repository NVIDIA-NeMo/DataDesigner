# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cross-record validation for collection lifecycle and inputs."""

from __future__ import annotations

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.collection_records import CollectionResult, CollectionState, CollectionStatus
from data_designer.slurm.state.outputs import CandidateOutputManifest, CollectionPlan
from data_designer.slurm.state.scheduler import SchedulerState
from data_designer.slurm.state.validation import StateContractError

_TERMINAL_COLLECTION_STATES = frozenset({CollectionState.SUCCEEDED, CollectionState.FAILED})


def validate_collection_inputs(
    resolved_plan: ResolvedSlurmRunPlan,
    collection_plan: CollectionPlan,
    candidates: tuple[CandidateOutputManifest, ...],
) -> tuple[CandidateOutputManifest, ...]:
    """Require one compatible, complete candidate for every planned shard."""
    _require(len(candidates) == len(collection_plan.planned_shards), "collection candidate set is incomplete")
    expected_shards = tuple(shard.shard_id for shard in collection_plan.planned_shards)
    actual_shards = tuple(candidate.shard_id for candidate in candidates)
    _require(actual_shards == expected_shards, "collection candidates do not match ordered planned shards")
    _require(all(candidate.winner_eligible for candidate in candidates), "collection candidate is not complete")

    schema_digests = {candidate.dataset_schema_digest for candidate in candidates}
    provenance_digests = {candidate.provenance_digest for candidate in candidates}
    _require(len(schema_digests) == 1, "collection candidates have incompatible schemas")
    _require(len(provenance_digests) == 1, "collection candidates have incompatible provenance")

    expected_records = resolved_plan.invocation.authored.num_records
    actual_records = sum(candidate.actual_records for candidate in candidates)
    _require(actual_records == expected_records, "collection candidate rows do not match requested run rows")
    return candidates


def validate_collection_status_transition(
    previous: CollectionStatus,
    current: CollectionStatus,
) -> CollectionStatus:
    """Validate immutable identity, monotonic revisions, and terminal evidence."""
    _require(current.collection_id == previous.collection_id, "collection identity cannot change")
    _require(current.run_id == previous.run_id, "collection run identity cannot change")
    _require(current.collection_plan == previous.collection_plan, "collection plan identity cannot change")
    _require(current.staging_directory == previous.staging_directory, "collection staging identity cannot change")
    _require(previous.state not in _TERMINAL_COLLECTION_STATES, "terminal collection status is immutable")
    _require(current.revision == previous.revision + 1, "collection status revision must increase by one")
    _require(current.updated_at >= previous.updated_at, "collection status timestamp cannot move backward")
    if previous.scheduler is not None:
        _require(current.scheduler == previous.scheduler, "collection scheduler identity cannot change")
    if previous.scheduler_observation is not None and current.scheduler_observation is not None:
        _require(
            current.scheduler_observation.observed_at >= previous.scheduler_observation.observed_at,
            "collection scheduler observation cannot move backward",
        )
    return current


def validate_collection_result(
    collection_plan: CollectionPlan,
    result: CollectionResult,
    *,
    expected_records: int,
    output_format: str,
) -> CollectionResult:
    """Bind a collected result to its immutable plan and exact row count."""
    _require(result.collection_id == collection_plan.collection_id, "collection result identity does not match")
    _require(result.run_id == collection_plan.run_id, "collection result run identity does not match")
    _require(
        result.collection_plan_sha256 == collection_plan.compute_sha256(),
        "collection result does not bind the collection plan",
    )
    _require(result.completed_at >= collection_plan.created_at, "collection completion precedes plan creation")
    _require(result.actual_records == expected_records, "collected result has the wrong row count")
    _require(len(result.files) == collection_plan.num_partitions, "collected result has the wrong partition count")
    suffix = "jsonl" if output_format == "jsonl" else output_format
    expected_paths = tuple(f"part-{index:05d}.{suffix}" for index in range(collection_plan.num_partitions))
    _require(
        tuple(output.relative_path for output in result.files) == expected_paths,
        "collected result files do not match deterministic partition intent",
    )
    return result


def derive_collection_state(observation_state: SchedulerState) -> CollectionState:
    """Map normalized scheduler evidence to collection lifecycle state."""
    if observation_state is SchedulerState.PENDING:
        return CollectionState.PENDING
    if observation_state is SchedulerState.RUNNING:
        return CollectionState.RUNNING
    if observation_state is SchedulerState.ACCOUNTING_LAG:
        return CollectionState.ACCOUNTING_LAG
    if observation_state is SchedulerState.UNKNOWN:
        return CollectionState.UNKNOWN
    return CollectionState.FAILED


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise StateContractError(message)


__all__ = [
    "derive_collection_state",
    "validate_collection_inputs",
    "validate_collection_result",
    "validate_collection_status_transition",
]
