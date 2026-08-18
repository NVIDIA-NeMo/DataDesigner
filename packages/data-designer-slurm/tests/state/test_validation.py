# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import pytest

from data_designer.slurm.state import (
    ArtifactReference,
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    AttemptTerminalClassification,
    CandidateOutputManifest,
    CollectionPlan,
    EffectiveAttemptState,
    EndpointPublicationState,
    ReadinessState,
    RunManifest,
    SchedulerObservation,
    SchedulerState,
    ShardManifest,
    ShardWinner,
    StateContractError,
    StateRecord,
    reconcile_attempt_observation,
    validate_attempt_manifest,
    validate_collection_plan,
    validate_readiness_transition,
    validate_shard_manifest,
    validate_shard_set,
    validate_shard_winner,
)

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
RecordT = TypeVar("RecordT", bound=StateRecord)


def _load(model: type[RecordT], filename: str) -> RecordT:
    return model.model_validate_json((GOLDEN_DIRECTORY / filename).read_text())


def test_run_shard_attempt_and_winner_contracts_match() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    attempt = _load(AttemptManifest, "successful_attempt.json")
    candidate = _load(CandidateOutputManifest, "candidate_output.json")
    winner = _load(ShardWinner, "shard_winner.json")

    assert validate_shard_manifest(run, shard) is shard
    assert validate_attempt_manifest(run, shard, attempt) is attempt
    assert validate_shard_winner(run, shard, attempt, candidate, winner) is winner


def test_cross_record_identity_and_digest_mismatches_fail() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    attempt = _load(AttemptManifest, "successful_attempt.json")
    candidate = _load(CandidateOutputManifest, "candidate_output.json")
    winner = _load(ShardWinner, "shard_winner.json")

    with pytest.raises(StateContractError, match="run_id"):
        validate_shard_manifest(run, shard.model_copy(update={"run_id": "another-run"}))

    wrong_reference = ArtifactReference(
        path=winner.candidate_manifest.path,
        sha256="0" * 64,
    )
    wrong_winner = winner.model_copy(update={"candidate_manifest": wrong_reference})
    with pytest.raises(StateContractError, match="candidate digest"):
        validate_shard_winner(run, shard, attempt, candidate, wrong_winner)

    wrong_count = candidate.model_copy(update={"requested_records": 101, "require_exact_record_count": False})
    with pytest.raises(StateContractError, match="record count"):
        validate_shard_winner(run, shard, attempt, wrong_count, winner)


def test_failed_partial_and_existing_winners_are_rejected() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    attempt = _load(AttemptManifest, "successful_attempt.json")
    candidate = _load(CandidateOutputManifest, "candidate_output.json")
    winner = _load(ShardWinner, "shard_winner.json")

    failed_attempt = attempt.model_copy(
        update={
            "state": AttemptLifecycleState.FAILED,
            "terminal_classification": AttemptTerminalClassification.NODE_FAILED,
        }
    )
    with pytest.raises(StateContractError, match="successful attempts"):
        validate_shard_winner(run, shard, failed_attempt, candidate, winner)

    partial_payload = json.loads((GOLDEN_DIRECTORY / "candidate_output.json").read_text())
    partial_payload.update({"actual_records": 99, "outcome": "partial"})
    partial_payload["files"][0]["record_count"] = 99
    partial = CandidateOutputManifest.model_validate_json(json.dumps(partial_payload))
    with pytest.raises(StateContractError, match="winner policy"):
        validate_shard_winner(run, shard, attempt, partial, winner)

    with pytest.raises(StateContractError, match="immutable"):
        validate_shard_winner(
            run,
            shard,
            attempt,
            candidate,
            winner,
            existing_winner=winner,
        )


def test_readiness_revisions_are_monotonic_and_preserve_authored_order() -> None:
    current = _load(AttemptReadiness, "single_node_readiness.json")
    starting_deployment = current.deployments[0].model_copy(
        update={
            "state": ReadinessState.STARTING,
            "ready_backends": 0,
            "endpoint_publication": EndpointPublicationState.PENDING,
        }
    )
    previous = current.model_copy(
        update={
            "revision": 2,
            "updated_at": datetime(2026, 8, 18, 12, 1, tzinfo=timezone.utc),
            "state": ReadinessState.STARTING,
            "deployments": (starting_deployment,),
        }
    )

    assert validate_readiness_transition(previous, current) is current
    with pytest.raises(StateContractError, match="revision"):
        validate_readiness_transition(previous, current.model_copy(update={"revision": 2}))

    multi = _load(AttemptReadiness, "multi_node_readiness.json")
    reordered = multi.model_copy(
        update={
            "revision": 5,
            "deployments": tuple(reversed(multi.deployments)),
        }
    )
    with pytest.raises(StateContractError, match="order or name"):
        validate_readiness_transition(multi, reordered)


def test_readiness_cannot_move_backward_or_change_backend_count() -> None:
    ready = _load(AttemptReadiness, "single_node_readiness.json")
    starting_deployment = ready.deployments[0].model_copy(
        update={
            "state": ReadinessState.STARTING,
            "ready_backends": 0,
            "endpoint_publication": EndpointPublicationState.PENDING,
        }
    )
    backward = ready.model_copy(
        update={
            "revision": 4,
            "state": ReadinessState.STARTING,
            "deployments": (starting_deployment,),
        }
    )
    with pytest.raises(StateContractError, match="cannot move"):
        validate_readiness_transition(ready, backward)

    changed_count = ready.model_copy(
        update={
            "revision": 4,
            "deployments": (ready.deployments[0].model_copy(update={"expected_backends": 2}),),
        }
    )
    with pytest.raises(StateContractError, match="backend count"):
        validate_readiness_transition(ready, changed_count)

    published_starting = ready.model_copy(
        update={
            "state": ReadinessState.STARTING,
            "deployments": (ready.deployments[0].model_copy(update={"state": ReadinessState.STARTING}),),
        }
    )
    unpublished = published_starting.model_copy(
        update={
            "revision": published_starting.revision + 1,
            "deployments": (
                published_starting.deployments[0].model_copy(
                    update={"endpoint_publication": EndpointPublicationState.PENDING}
                ),
            ),
        }
    )
    with pytest.raises(StateContractError, match="endpoint publication"):
        validate_readiness_transition(published_starting, unpublished)


def test_collection_requires_exact_winner_set_and_digests() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    plan = _load(CollectionPlan, "collection_plan.json")
    winner = _load(ShardWinner, "shard_winner.json")

    assert validate_shard_set(run, (shard,)) == (shard,)
    assert validate_collection_plan(run, plan, (shard,), (winner,)) is plan
    with pytest.raises(StateContractError, match="winner set"):
        validate_collection_plan(run, plan, (shard,), ())

    wrong_reference = ArtifactReference(
        path=plan.planned_shards[0].winner_manifest.path,
        sha256="0" * 64,
    )
    wrong_planned_shard = plan.planned_shards[0].model_copy(update={"winner_manifest": wrong_reference})
    wrong_plan = plan.model_copy(update={"planned_shards": (wrong_planned_shard,)})
    with pytest.raises(StateContractError, match="digest mismatch"):
        validate_collection_plan(run, wrong_plan, (shard,), (winner,))


def test_collection_rejects_an_invented_shard_even_when_its_digest_matches() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    plan = _load(CollectionPlan, "collection_plan.json")
    winner = _load(ShardWinner, "shard_winner.json").model_copy(update={"shard_id": "invented-shard"})
    winner_reference = ArtifactReference(
        path=plan.planned_shards[0].winner_manifest.path,
        sha256=winner.compute_sha256(),
    )
    planned_shard = plan.planned_shards[0].model_copy(
        update={"shard_id": "invented-shard", "winner_manifest": winner_reference}
    )
    altered_plan = plan.model_copy(update={"planned_shards": (planned_shard,)})

    with pytest.raises(StateContractError, match="planned shards"):
        validate_collection_plan(run, altered_plan, (shard,), (winner,))


def test_terminal_attempt_evidence_overrides_stale_readiness() -> None:
    attempt = _load(AttemptManifest, "failed_attempt.json")
    readiness = _load(AttemptReadiness, "stale_readiness.json")
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 13, 5, tzinfo=timezone.utc),
        state=SchedulerState.RUNNING,
    )

    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 13, 5, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.FAILED
    )


def test_terminal_scheduler_failure_overrides_successful_attempt() -> None:
    attempt = _load(AttemptManifest, "successful_attempt.json")
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        state=SchedulerState.NODE_FAILED,
    )

    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.FAILED
    )


def test_accounting_lag_is_nonterminal_until_its_deadline() -> None:
    attempt = _load(AttemptManifest, "successful_attempt.json").model_copy(
        update={
            "state": AttemptLifecycleState.RUNNING,
            "terminal_classification": None,
            "candidate_output": None,
        }
    )
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    scheduler = _load(SchedulerObservation, "accounting_lag.json")

    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 7, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.ACCOUNTING_LAG
    )
    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 11, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.UNKNOWN
    )


def test_readiness_never_declares_success() -> None:
    attempt = _load(AttemptManifest, "successful_attempt.json").model_copy(
        update={
            "state": AttemptLifecycleState.RUNNING,
            "terminal_classification": None,
            "candidate_output": None,
        }
    )
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 2, tzinfo=timezone.utc),
        state=SchedulerState.UNKNOWN,
    )

    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 2, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.RUNNING
    )
