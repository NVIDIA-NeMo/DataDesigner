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
    StateValue,
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
ValueT = TypeVar("ValueT", bound=StateValue)


def _load(model: type[RecordT], filename: str) -> RecordT:
    return model.model_validate_json((GOLDEN_DIRECTORY / filename).read_text())


def _json_value(value: object) -> object:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, StateValue):
        return value.model_dump(mode="json")
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    return value


def _validated_copy(record: ValueT, **updates: object) -> ValueT:
    payload = record.model_dump(mode="json")
    payload.update({key: _json_value(value) for key, value in updates.items()})
    return type(record).model_validate_json(json.dumps(payload))


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
        validate_shard_manifest(run, _validated_copy(shard, run_id="another-run"))

    wrong_reference = ArtifactReference(
        path=winner.candidate_manifest.path,
        sha256="0" * 64,
    )
    wrong_winner = _validated_copy(winner, candidate_manifest=wrong_reference)
    with pytest.raises(StateContractError, match="candidate digest"):
        validate_shard_winner(run, shard, attempt, candidate, wrong_winner)

    wrong_range = _validated_copy(
        shard.record_range,
        end_index_exclusive=shard.record_range.end_index_exclusive + 1,
    )
    wrong_shard = _validated_copy(shard, record_range=wrong_range)
    with pytest.raises(StateContractError, match="record count"):
        validate_shard_winner(run, wrong_shard, attempt, candidate, winner)


def test_shard_set_rejects_overlapping_ranges_and_shared_resume_workspace() -> None:
    run = _validated_copy(_load(RunManifest, "run_manifest.json"), shard_count=2)
    first = _load(ShardManifest, "shard_manifest.json")
    second_range = _validated_copy(
        first.record_range,
        start_index=first.record_range.end_index_exclusive,
        end_index_exclusive=first.record_range.end_index_exclusive + first.record_range.record_count,
    )
    second = _validated_copy(
        first,
        shard_id="shard-0001",
        shard_index=1,
        record_range=second_range,
        resume_workspace_id="resume-shard-0001",
    )

    assert validate_shard_set(run, (first, second)) == (first, second)

    overlapping_range = _validated_copy(
        second.record_range,
        start_index=first.record_range.end_index_exclusive - 1,
    )
    overlapping = _validated_copy(second, record_range=overlapping_range)
    with pytest.raises(StateContractError, match="must not overlap"):
        validate_shard_set(run, (first, overlapping))

    shared_workspace = _validated_copy(second, resume_workspace_id=first.resume_workspace_id)
    with pytest.raises(StateContractError, match="workspace IDs must be unique"):
        validate_shard_set(run, (first, shared_workspace))


def test_failed_partial_and_existing_winners_are_rejected() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    attempt = _load(AttemptManifest, "successful_attempt.json")
    candidate = _load(CandidateOutputManifest, "candidate_output.json")
    winner = _load(ShardWinner, "shard_winner.json")

    failed_attempt = _validated_copy(
        attempt,
        state=AttemptLifecycleState.FAILED,
        terminal_classification=AttemptTerminalClassification.NODE_FAILED,
    )
    with pytest.raises(StateContractError, match="successful attempts"):
        validate_shard_winner(run, shard, failed_attempt, candidate, winner)

    partial_payload = json.loads((GOLDEN_DIRECTORY / "candidate_output.json").read_text())
    partial_payload.update(
        {
            "actual_records": 99,
            "outcome": "partial",
        }
    )
    partial_payload["files"][0]["record_count"] = 99
    partial = CandidateOutputManifest.model_validate_json(json.dumps(partial_payload))
    partial_reference = ArtifactReference(
        path=winner.candidate_manifest.path,
        sha256=partial.compute_sha256(),
    )
    partial_attempt = _validated_copy(attempt, candidate_output=partial_reference)
    partial_winner = _validated_copy(winner, candidate_manifest=partial_reference)
    with pytest.raises(StateContractError, match="winner policy"):
        validate_shard_winner(run, shard, partial_attempt, partial, partial_winner)

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
    starting_deployment = _validated_copy(
        current.deployments[0],
        state=ReadinessState.STARTING,
        ready_backends=0,
        endpoint_publication=EndpointPublicationState.PENDING,
        last_probe=None,
    )
    previous = _validated_copy(
        current,
        revision=2,
        updated_at=datetime(2026, 8, 18, 12, 1, tzinfo=timezone.utc),
        state=ReadinessState.STARTING,
        deployments=(starting_deployment,),
    )

    assert validate_readiness_transition(previous, current) is current
    with pytest.raises(StateContractError, match="revision"):
        validate_readiness_transition(previous, _validated_copy(current, revision=2))

    multi = _load(AttemptReadiness, "multi_node_readiness.json")
    reordered = _validated_copy(
        multi,
        revision=5,
        deployments=tuple(reversed(multi.deployments)),
    )
    with pytest.raises(StateContractError, match="order or name"):
        validate_readiness_transition(multi, reordered)


def test_readiness_cannot_move_backward_or_change_backend_count() -> None:
    ready = _load(AttemptReadiness, "single_node_readiness.json")
    starting_deployment = _validated_copy(
        ready.deployments[0],
        state=ReadinessState.STARTING,
        ready_backends=0,
        endpoint_publication=EndpointPublicationState.PENDING,
    )
    backward = _validated_copy(
        ready,
        revision=4,
        state=ReadinessState.STARTING,
        deployments=(starting_deployment,),
    )
    with pytest.raises(StateContractError, match="cannot move"):
        validate_readiness_transition(ready, backward)

    changed_deployment = _validated_copy(
        ready.deployments[0],
        expected_backends=2,
        ready_backends=2,
    )
    changed_count = _validated_copy(
        ready,
        revision=4,
        deployments=(changed_deployment,),
    )
    with pytest.raises(StateContractError, match="backend count"):
        validate_readiness_transition(ready, changed_count)

    stopped_deployment = _validated_copy(
        ready.deployments[0],
        state=ReadinessState.STOPPED,
        ready_backends=0,
        endpoint_publication=EndpointPublicationState.PENDING,
    )
    unpublished = _validated_copy(
        ready,
        revision=ready.revision + 1,
        state=ReadinessState.STOPPED,
        deployments=(stopped_deployment,),
    )
    with pytest.raises(StateContractError, match="endpoint publication"):
        validate_readiness_transition(ready, unpublished)


def test_readiness_probe_evidence_cannot_regress_or_disappear() -> None:
    previous = _load(AttemptReadiness, "single_node_readiness.json")
    previous_probe = previous.deployments[0].last_probe
    assert previous_probe is not None

    regressed_probe = _validated_copy(
        previous_probe,
        observed_at=datetime(2026, 8, 18, 12, 1, tzinfo=timezone.utc),
    )
    regressed_deployment = _validated_copy(previous.deployments[0], last_probe=regressed_probe)
    regressed = _validated_copy(
        previous,
        revision=previous.revision + 1,
        updated_at=datetime(2026, 8, 18, 12, 3, tzinfo=timezone.utc),
        deployments=(regressed_deployment,),
    )
    with pytest.raises(StateContractError, match="probe observation cannot move backward"):
        validate_readiness_transition(previous, regressed)

    removed_deployment = _validated_copy(previous.deployments[0], last_probe=None)
    removed = _validated_copy(
        previous,
        revision=previous.revision + 1,
        updated_at=datetime(2026, 8, 18, 12, 3, tzinfo=timezone.utc),
        deployments=(removed_deployment,),
    )
    with pytest.raises(StateContractError, match="probe evidence cannot be removed"):
        validate_readiness_transition(previous, removed)


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
    wrong_planned_shard = _validated_copy(plan.planned_shards[0], winner_manifest=wrong_reference)
    wrong_plan = _validated_copy(plan, planned_shards=(wrong_planned_shard,))
    with pytest.raises(StateContractError, match="digest mismatch"):
        validate_collection_plan(run, wrong_plan, (shard,), (winner,))


def test_collection_rejects_an_invented_shard_even_when_its_digest_matches() -> None:
    run = _load(RunManifest, "run_manifest.json")
    shard = _load(ShardManifest, "shard_manifest.json")
    plan = _load(CollectionPlan, "collection_plan.json")
    winner = _validated_copy(_load(ShardWinner, "shard_winner.json"), shard_id="invented-shard")
    winner_reference = ArtifactReference(
        path=plan.planned_shards[0].winner_manifest.path,
        sha256=winner.compute_sha256(),
    )
    planned_shard = _validated_copy(
        plan.planned_shards[0],
        shard_id="invented-shard",
        winner_manifest=winner_reference,
    )
    altered_plan = _validated_copy(plan, planned_shards=(planned_shard,))

    with pytest.raises(StateContractError, match="planned shards"):
        validate_collection_plan(run, altered_plan, (shard,), (winner,))


def test_terminal_attempt_evidence_overrides_stale_readiness() -> None:
    attempt = _load(AttemptManifest, "failed_attempt.json")
    readiness = _load(AttemptReadiness, "stale_readiness.json")
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        schema_version=1,
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
        schema_version=1,
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


@pytest.mark.parametrize("scheduler_state", (SchedulerState.PENDING, SchedulerState.RUNNING))
def test_failed_readiness_overrides_nonterminal_scheduler_state(scheduler_state: SchedulerState) -> None:
    attempt = _validated_copy(
        _load(AttemptManifest, "successful_attempt.json"),
        state=AttemptLifecycleState.RUNNING,
        terminal_classification=None,
        candidate_output=None,
    )
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    failed_deployment = _validated_copy(
        readiness.deployments[0],
        state=ReadinessState.FAILED,
        ready_backends=0,
        endpoint_publication=EndpointPublicationState.FAILED,
    )
    failed_readiness = _validated_copy(
        readiness,
        state=ReadinessState.FAILED,
        deployments=(failed_deployment,),
    )
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        schema_version=1,
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        state=scheduler_state,
    )

    assert (
        reconcile_attempt_observation(
            attempt,
            failed_readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.FAILED
    )


def test_reconciliation_rejects_stale_or_future_evidence() -> None:
    attempt = _load(AttemptManifest, "successful_attempt.json")
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    assert attempt.scheduler is not None

    stale_scheduler = SchedulerObservation(
        schema_version=1,
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 0, 1, tzinfo=timezone.utc),
        state=SchedulerState.NODE_FAILED,
    )
    with pytest.raises(StateContractError, match="cannot precede attempt creation"):
        reconcile_attempt_observation(
            attempt,
            readiness,
            stale_scheduler,
            current_time=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        )

    current_scheduler = SchedulerObservation(
        schema_version=1,
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        state=SchedulerState.UNKNOWN,
    )
    future_readiness = _validated_copy(
        readiness,
        updated_at=datetime(2026, 8, 18, 12, 7, tzinfo=timezone.utc),
    )
    with pytest.raises(StateContractError, match="cannot precede readiness update"):
        reconcile_attempt_observation(
            attempt,
            future_readiness,
            current_scheduler,
            current_time=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        )


def test_reconciliation_covers_nonterminal_and_fallback_states() -> None:
    successful_attempt = _load(AttemptManifest, "successful_attempt.json")
    attempt = _validated_copy(
        successful_attempt,
        state=AttemptLifecycleState.RUNNING,
        terminal_classification=None,
        candidate_output=None,
    )
    ready = _load(AttemptReadiness, "single_node_readiness.json")
    pending_deployment = _validated_copy(
        ready.deployments[0],
        state=ReadinessState.PENDING,
        ready_backends=0,
        endpoint_publication=EndpointPublicationState.PENDING,
        last_probe=None,
    )
    pending = _validated_copy(ready, state=ReadinessState.PENDING, deployments=(pending_deployment,))
    stopped_deployment = _validated_copy(
        ready.deployments[0],
        state=ReadinessState.STOPPED,
        ready_backends=0,
    )
    stopped = _validated_copy(ready, state=ReadinessState.STOPPED, deployments=(stopped_deployment,))
    assert attempt.scheduler is not None
    observed_at = datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc)
    current_time = observed_at

    cases = (
        (attempt, ready, SchedulerState.PENDING, EffectiveAttemptState.PENDING),
        (attempt, ready, SchedulerState.RUNNING, EffectiveAttemptState.RUNNING),
        (attempt, ready, SchedulerState.COMPLETED, EffectiveAttemptState.FAILED),
        (attempt, pending, SchedulerState.UNKNOWN, EffectiveAttemptState.PENDING),
        (attempt, stopped, SchedulerState.UNKNOWN, EffectiveAttemptState.UNKNOWN),
        (successful_attempt, ready, SchedulerState.UNKNOWN, EffectiveAttemptState.SUCCEEDED),
    )
    for case_attempt, case_readiness, scheduler_state, expected in cases:
        scheduler = SchedulerObservation(
            schema_version=1,
            scheduler=attempt.scheduler,
            observed_at=observed_at,
            state=scheduler_state,
        )
        assert (
            reconcile_attempt_observation(
                case_attempt,
                case_readiness,
                scheduler,
                current_time=current_time,
            )
            is expected
        )


def test_accounting_lag_is_nonterminal_until_its_deadline() -> None:
    attempt = _validated_copy(
        _load(AttemptManifest, "successful_attempt.json"),
        state=AttemptLifecycleState.RUNNING,
        terminal_classification=None,
        candidate_output=None,
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
    attempt = _validated_copy(
        _load(AttemptManifest, "successful_attempt.json"),
        state=AttemptLifecycleState.RUNNING,
        terminal_classification=None,
        candidate_output=None,
    )
    readiness = _load(AttemptReadiness, "single_node_readiness.json")
    assert attempt.scheduler is not None
    scheduler = SchedulerObservation(
        schema_version=1,
        scheduler=attempt.scheduler,
        observed_at=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        state=SchedulerState.UNKNOWN,
    )

    assert (
        reconcile_attempt_observation(
            attempt,
            readiness,
            scheduler,
            current_time=datetime(2026, 8, 18, 12, 6, tzinfo=timezone.utc),
        )
        is EffectiveAttemptState.RUNNING
    )
