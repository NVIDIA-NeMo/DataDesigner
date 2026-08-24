# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import posixpath
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TypeVar

import pytest

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ArtifactReference, ContractValue, RecordRange
from data_designer.slurm.integration import (
    IntegrationContractError,
    PlanStateValidator,
    validate_finalization_chain,
    validate_initial_readiness,
    validate_plan_shards,
    validate_planned_attempt,
)
from data_designer.slurm.planning import (
    ArtifactReference as PlanningArtifactReference,
)
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    ArtifactReference as StateArtifactReference,
)
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    AttemptTerminalClassification,
    CandidateOutputManifest,
    DeploymentReadiness,
    EndpointPublicationState,
    ReadinessState,
    RunManifest,
    SchedulerIdentity,
    ShardManifest,
    ShardWinner,
)

TEST_ROOT = Path(__file__).parents[1]
CONTRACT_GOLDEN_DIR = TEST_ROOT / "contracts" / "golden"
INTEGRATION_GOLDEN_DIR = Path(__file__).parent / "golden"
CREATED_AT = datetime(2026, 8, 19, 12, 0, tzinfo=UTC)
_RecordT = TypeVar("_RecordT", bound=ContractValue)


@dataclass(frozen=True)
class IntegrationRecords:
    plan: ResolvedSlurmRunPlan
    validator: PlanStateValidator
    run: RunManifest
    shards: tuple[ShardManifest, ...]
    attempt: AttemptManifest
    readiness: AttemptReadiness
    client_result: ClientResult
    candidate: CandidateOutputManifest
    winner: ShardWinner


@pytest.fixture
def records() -> IntegrationRecords:
    plan = _load_plan("single_node_plan.json")
    payload = json.loads((INTEGRATION_GOLDEN_DIR / "finalization_chain.json").read_text())
    return IntegrationRecords(
        plan=plan,
        validator=PlanStateValidator(plan),
        run=_load_record(RunManifest, payload["run"]),
        shards=tuple(_load_record(ShardManifest, shard) for shard in payload["shards"]),
        attempt=_load_record(AttemptManifest, payload["attempt"]),
        readiness=_load_record(AttemptReadiness, payload["readiness"]),
        client_result=_load_record(ClientResult, payload["client_result"]),
        candidate=_load_record(CandidateOutputManifest, payload["candidate"]),
        winner=_load_record(ShardWinner, payload["winner"]),
    )


def test_shared_contract_types_retain_exact_identity() -> None:
    assert PlanningArtifactReference is ArtifactReference
    assert StateArtifactReference is ArtifactReference


def test_golden_records_validate_every_plan_state_join(records: IntegrationRecords) -> None:
    planned_shard = records.plan.shards[0]

    assert records.validator.validate_plan_shards(records.run, records.shards) is records.shards
    assert records.validator.validate_initial_readiness(records.attempt, records.readiness) is records.readiness
    assert records.validator.validate_planned_attempt(planned_shard, records.attempt) is records.attempt
    assert (
        records.validator.validate_finalization_chain(
            planned_shard,
            records.attempt,
            records.client_result,
            records.candidate,
            records.winner,
        )
        is records.winner
    )


def test_plan_shards_reject_missing_extra_and_mismatched_state(records: IntegrationRecords) -> None:
    with pytest.raises(IntegrationContractError, match="exactly the run shard count"):
        validate_plan_shards(records.plan, records.run, ())
    with pytest.raises(IntegrationContractError, match="exactly the run shard count"):
        validate_plan_shards(records.plan, records.run, records.shards + records.shards)

    mismatched = records.shards[0].model_copy(
        update={"record_range": RecordRange(start_index=1, end_index_exclusive=8)}
    )
    with pytest.raises(IntegrationContractError, match="record range"):
        validate_plan_shards(records.plan, records.run, (mismatched,))


def test_plan_shards_reject_reordered_state() -> None:
    plan = _load_plan("multi_node_plan.json")
    run, shards = _state_shards_for_plan(plan)

    with pytest.raises(IntegrationContractError, match="ordered"):
        validate_plan_shards(plan, run, tuple(reversed(shards)))


def test_initial_readiness_rejects_plan_order_alias_and_backend_count(records: IntegrationRecords) -> None:
    deployment = records.readiness.deployments[0]
    wrong_alias = deployment.model_copy(update={"model_alias": "other"})
    readiness = records.readiness.model_copy(update={"deployments": (wrong_alias,)})
    with pytest.raises(IntegrationContractError, match="deployments"):
        validate_initial_readiness(records.plan, records.attempt, readiness)

    wrong_count = deployment.model_copy(update={"expected_backends": 2})
    readiness = records.readiness.model_copy(update={"deployments": (wrong_count,)})
    with pytest.raises(IntegrationContractError, match="deployments"):
        validate_initial_readiness(records.plan, records.attempt, readiness)

    multi_plan = _load_plan("multi_node_plan.json")
    attempt = _attempt_for_plan(multi_plan)
    readiness = _pending_readiness(multi_plan, attempt)
    reordered = readiness.model_copy(update={"deployments": tuple(reversed(readiness.deployments))})
    with pytest.raises(IntegrationContractError, match="deployments"):
        validate_initial_readiness(multi_plan, attempt, reordered)


def test_initial_readiness_requires_first_pending_revision(records: IntegrationRecords) -> None:
    with pytest.raises(IntegrationContractError, match="revision 1"):
        validate_initial_readiness(
            records.plan,
            records.attempt,
            records.readiness.model_copy(update={"revision": 2}),
        )
    with pytest.raises(IntegrationContractError, match="must be pending"):
        validate_initial_readiness(
            records.plan,
            records.attempt,
            records.readiness.model_copy(update={"state": ReadinessState.READY}),
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("unplanned_shard", "planned shard"),
        ("wrong_array_task", "array task"),
        ("nondeterministic_attempt_id", "ordinal"),
        ("unsubmitted_attempt", "scheduler"),
        ("unsubmitted_attempt_with_scheduler", "submitted"),
    ],
)
def test_initial_readiness_rejects_invalid_planned_attempt(
    records: IntegrationRecords,
    mutation: str,
    message: str,
) -> None:
    attempt = records.attempt
    readiness = records.readiness
    if mutation == "unplanned_shard":
        attempt = attempt.model_copy(update={"shard_id": "shard-99999"})
        readiness = readiness.model_copy(update={"shard_id": attempt.shard_id})
    elif mutation == "wrong_array_task":
        attempt = attempt.model_copy(update={"scheduler": SchedulerIdentity(array_job_id=4101, array_task_id=1)})
    elif mutation == "nondeterministic_attempt_id":
        attempt = attempt.model_copy(update={"attempt_id": "attempt-0002"})
        readiness = readiness.model_copy(update={"attempt_id": attempt.attempt_id})
    elif mutation == "unsubmitted_attempt":
        attempt = attempt.model_copy(
            update={
                "state": AttemptLifecycleState.CREATED,
                "scheduler": None,
                "terminal_classification": None,
                "candidate_output": None,
            }
        )
    else:
        attempt = AttemptManifest.model_validate(
            attempt.model_dump(mode="python")
            | {
                "state": AttemptLifecycleState.CREATED,
                "terminal_classification": None,
                "candidate_output": None,
            }
        )

    with pytest.raises(IntegrationContractError, match=message):
        validate_initial_readiness(records.plan, attempt, readiness)


def test_planned_attempt_rejects_task_identity_and_unsubmitted_state(records: IntegrationRecords) -> None:
    planned_shard = records.plan.shards[0]
    wrong_task = records.attempt.model_copy(update={"scheduler": SchedulerIdentity(array_job_id=4101, array_task_id=1)})
    with pytest.raises(IntegrationContractError, match="array task"):
        validate_planned_attempt(records.plan, planned_shard, wrong_task)

    wrong_id = records.attempt.model_copy(update={"attempt_id": "attempt-0002"})
    with pytest.raises(IntegrationContractError, match="ordinal"):
        validate_planned_attempt(records.plan, planned_shard, wrong_id)

    no_scheduler = records.attempt.model_copy(
        update={
            "state": AttemptLifecycleState.CREATED,
            "scheduler": None,
            "terminal_classification": None,
            "candidate_output": None,
        }
    )
    with pytest.raises(IntegrationContractError, match="scheduler"):
        validate_planned_attempt(records.plan, planned_shard, no_scheduler)

    created_with_scheduler = AttemptManifest.model_validate(
        no_scheduler.model_dump(mode="python") | {"scheduler": records.attempt.scheduler}
    )
    with pytest.raises(IntegrationContractError, match="submitted"):
        validate_planned_attempt(records.plan, planned_shard, created_with_scheduler)


def test_plan_state_validator_reuses_one_context_for_an_attempt_batch() -> None:
    plan = _load_plan("multi_node_plan.json")
    validator = PlanStateValidator(plan)
    attempts = tuple(_attempt_for_planned_shard(plan, planned_shard) for planned_shard in plan.shards)

    assert (
        tuple(
            validator.validate_planned_attempt(planned_shard, attempt)
            for planned_shard, attempt in zip(plan.shards, attempts, strict=True)
        )
        == attempts
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("partial_result", "complete client results"),
        ("failed_result", "complete client results"),
        ("failed_attempt", "successful attempts"),
        ("stale_winner", "winner attempt_id"),
        ("digest_mismatch", "digest"),
        ("dataset_path_mismatch", "dataset path"),
        ("client_before_candidate", "client completion"),
        ("winner_before_attempt", "winner publication"),
    ],
)
def test_finalization_rejects_invalid_chains(
    records: IntegrationRecords,
    mutation: str,
    message: str,
) -> None:
    attempt = records.attempt
    client_result = records.client_result
    candidate = records.candidate
    winner = records.winner
    if mutation == "partial_result":
        client_result = client_result.model_copy(
            update={"outcome": ClientOutcome.PARTIAL, "actual_records": 4, "early_shutdown": True}
        )
    elif mutation == "failed_result":
        client_result = ClientResult.model_validate(
            client_result.model_dump(mode="python")
            | {
                "actual_records": None,
                "outcome": ClientOutcome.FAILED,
                "dataset_path": None,
                "early_shutdown": None,
                "effective_resume_mode": None,
                "candidate_output_manifest": None,
                "error_code": "generation_failed",
                "redacted_message": "generation failed",
            }
        )
    elif mutation == "failed_attempt":
        attempt = AttemptManifest.model_validate(
            attempt.model_dump(mode="python")
            | {
                "state": AttemptLifecycleState.FAILED,
                "terminal_classification": AttemptTerminalClassification.FAILED,
                "candidate_output": None,
            }
        )
    elif mutation == "stale_winner":
        winner = winner.model_copy(update={"attempt_id": "attempt-0002"})
    elif mutation == "digest_mismatch":
        candidate = candidate.model_copy(update={"provenance_digest": "c" * 64})
    elif mutation == "dataset_path_mismatch":
        candidate = candidate.model_copy(update={"dataset_path": "/workspace/other/dataset"})
    elif mutation == "client_before_candidate":
        client_result = client_result.model_copy(update={"completed_at": candidate.created_at - timedelta(seconds=1)})
    else:
        winner = winner.model_copy(update={"published_at": records.attempt.updated_at - timedelta(seconds=1)})

    with pytest.raises(IntegrationContractError, match=message):
        validate_finalization_chain(
            records.plan,
            records.plan.shards[0],
            attempt,
            client_result,
            candidate,
            winner,
        )


def test_finalization_rejects_plan_reference_drift(records: IntegrationRecords) -> None:
    drifted = records.attempt.model_copy(
        update={
            "resolved_plan": ArtifactReference(
                path=records.attempt.resolved_plan.path,
                sha256="a" * 64,
            )
        }
    )

    with pytest.raises(IntegrationContractError, match="digest"):
        validate_planned_attempt(records.plan, records.plan.shards[0], drifted)


def test_finalization_accepts_effective_resume_always(records: IntegrationRecords) -> None:
    plan_payload = records.plan.model_dump(mode="python")
    plan_payload["invocation"]["authored"]["resume"] = "always"
    plan = ResolvedSlurmRunPlan.model_validate(plan_payload)
    planned_shard = plan.shards[0]
    candidate = CandidateOutputManifest.model_validate(
        records.candidate.model_dump(mode="python") | {"dataset_path": planned_shard.resume_workspace.path}
    )
    original_candidate_reference = records.client_result.candidate_output_manifest
    assert original_candidate_reference is not None
    candidate_reference = ArtifactReference(
        path=original_candidate_reference.path,
        sha256=candidate.compute_sha256(),
    )
    attempt = AttemptManifest.model_validate(
        records.attempt.model_dump(mode="python")
        | {"candidate_output": candidate_reference, "resolved_plan": _plan_reference(plan)}
    )
    client_result = ClientResult.model_validate(
        records.client_result.model_dump(mode="python")
        | {
            "requested_resume_mode": "always",
            "effective_resume_mode": "always",
            "dataset_path": planned_shard.resume_workspace.path,
            "candidate_output_manifest": candidate_reference,
        }
    )
    winner = ShardWinner.model_validate(
        records.winner.model_dump(mode="python") | {"candidate_manifest": candidate_reference}
    )

    assert (
        validate_finalization_chain(
            plan,
            planned_shard,
            attempt,
            client_result,
            candidate,
            winner,
        )
        is winner
    )


def test_integration_golden_contains_no_environment_specific_values() -> None:
    payload = (INTEGRATION_GOLDEN_DIR / "finalization_chain.json").read_text().casefold()

    for forbidden in ('"token":', '"password":', '"account":', '"partition":', "/users/", "/home/"):
        assert forbidden not in payload


def _load_plan(name: str) -> ResolvedSlurmRunPlan:
    return ResolvedSlurmRunPlan.model_validate_json((CONTRACT_GOLDEN_DIR / name).read_text())


def _load_record(record_type: type[_RecordT], payload: object) -> _RecordT:
    return record_type.model_validate_json(json.dumps(payload))


def _plan_reference(plan: ResolvedSlurmRunPlan) -> ArtifactReference:
    return ArtifactReference(
        path=posixpath.join(posixpath.dirname(plan.authored_config.path), "resolved-plan.json"),
        sha256=plan.compute_sha256(),
    )


def _state_shards_for_plan(plan: ResolvedSlurmRunPlan) -> tuple[RunManifest, tuple[ShardManifest, ...]]:
    run = RunManifest(
        schema_version=1,
        run_id=plan.run_id,
        created_at=CREATED_AT,
        authored_config=plan.authored_config,
        resolved_plan=_plan_reference(plan),
        shard_count=len(plan.shards),
    )
    shards = tuple(
        ShardManifest(
            schema_version=1,
            run_id=plan.run_id,
            shard_id=planned.shard_id,
            shard_index=planned.shard_index,
            record_range=planned.record_range,
            input_partition=planned.input_partition,
            resume_workspace=planned.resume_workspace,
            created_at=CREATED_AT + timedelta(seconds=index + 1),
        )
        for index, planned in enumerate(plan.shards)
    )
    return run, shards


def _attempt_for_plan(plan: ResolvedSlurmRunPlan) -> AttemptManifest:
    return _attempt_for_planned_shard(plan, plan.shards[0])


def _attempt_for_planned_shard(plan: ResolvedSlurmRunPlan, shard: PlannedShard) -> AttemptManifest:
    return AttemptManifest(
        schema_version=1,
        run_id=plan.run_id,
        shard_id=shard.shard_id,
        attempt_id="attempt-0001",
        attempt_ordinal=1,
        resolved_plan=_plan_reference(plan),
        state=AttemptLifecycleState.SUBMITTED,
        scheduler=SchedulerIdentity(array_job_id=4101, array_task_id=shard.array_task_index),
        created_at=CREATED_AT,
        updated_at=CREATED_AT,
    )


def _pending_readiness(plan: ResolvedSlurmRunPlan, attempt: AttemptManifest) -> AttemptReadiness:
    return AttemptReadiness(
        schema_version=1,
        run_id=attempt.run_id,
        shard_id=attempt.shard_id,
        attempt_id=attempt.attempt_id,
        revision=1,
        updated_at=attempt.created_at,
        state=ReadinessState.PENDING,
        deployments=tuple(
            DeploymentReadiness(
                deployment_id=deployment.deployment_id,
                model_alias=deployment.authored.model_alias,
                state=ReadinessState.PENDING,
                expected_backends=deployment.topology.replica_count,
                ready_backends=0,
                endpoint_publication=EndpointPublicationState.PENDING,
            )
            for deployment in plan.deployments
        ),
    )
