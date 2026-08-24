# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation across Slurm execution-plan and runtime-state contracts."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ArtifactReference, ShardId
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptReadiness,
    AttemptTerminalClassification,
    CandidateOutputManifest,
    ReadinessState,
    RunManifest,
    ShardManifest,
    ShardWinner,
    StateContractError,
    validate_shard_set,
)


class IntegrationContractError(ValueError):
    """Raised when reviewed plan and state records disagree."""


@dataclass(frozen=True, slots=True)
class PlanStateValidator:
    """Validate records against one resolved plan with reusable derived state."""

    plan: ResolvedSlurmRunPlan
    _plan_reference: ArtifactReference = field(init=False, repr=False)
    _shards_by_id: Mapping[ShardId, PlannedShard] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        run_root = posixpath.dirname(self.plan.authored_config.path)
        object.__setattr__(
            self,
            "_plan_reference",
            ArtifactReference(
                path=posixpath.join(run_root, "resolved-plan.json"),
                sha256=self.plan.compute_sha256(),
            ),
        )
        object.__setattr__(
            self,
            "_shards_by_id",
            MappingProxyType({planned_shard.shard_id: planned_shard for planned_shard in self.plan.shards}),
        )

    def validate_plan_shards(
        self,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> tuple[ShardManifest, ...]:
        """Validate the complete ordered state shard set against planned intent."""
        _require(run.run_id == self.plan.run_id, "run manifest identity does not match the resolved plan")
        _require(
            run.authored_config == self.plan.authored_config,
            "run authored config does not match the resolved plan",
        )
        self._validate_plan_reference(run.resolved_plan)
        _require(run.shard_count == len(self.plan.shards), "run shard count does not match the resolved plan")
        try:
            validate_shard_set(run, shards)
        except StateContractError as exc:
            raise IntegrationContractError(str(exc)) from exc

        _require(len(shards) == len(self.plan.shards), "state shards must exactly match the planned shard count")
        for planned, persisted in zip(self.plan.shards, shards, strict=True):
            _require(persisted.shard_id == planned.shard_id, "state shard identity does not match planned order")
            _require(persisted.shard_index == planned.shard_index, "state shard index does not match planned order")
            _require(persisted.record_range == planned.record_range, "state shard record range does not match the plan")
            _require(
                persisted.input_partition == planned.input_partition,
                "state shard input partition does not match the plan",
            )
            _require(
                persisted.resume_workspace == planned.resume_workspace,
                "state shard resume workspace does not match the plan",
            )
        return shards

    def validate_initial_readiness(
        self,
        attempt: AttemptManifest,
        readiness: AttemptReadiness,
    ) -> AttemptReadiness:
        """Anchor the first readiness snapshot to a fully validated planned attempt."""
        planned_shard = self._get_planned_shard(attempt.shard_id)
        self.validate_planned_attempt(planned_shard, attempt)
        _require(readiness.run_id == attempt.run_id, "readiness run_id does not match the attempt")
        _require(readiness.shard_id == attempt.shard_id, "readiness shard_id does not match the attempt")
        _require(readiness.attempt_id == attempt.attempt_id, "readiness attempt_id does not match the attempt")
        _require(readiness.revision == 1, "initial readiness must use revision 1")
        _require(readiness.state is ReadinessState.PENDING, "initial readiness must be pending")
        _require(readiness.updated_at >= attempt.created_at, "initial readiness cannot precede attempt creation")

        expected = tuple(
            (
                deployment.deployment_id,
                deployment.authored.model_alias,
                deployment.topology.replica_count,
            )
            for deployment in self.plan.deployments
        )
        actual = tuple(
            (deployment.deployment_id, deployment.model_alias, deployment.expected_backends)
            for deployment in readiness.deployments
        )
        _require(actual == expected, "initial readiness deployments do not match the resolved plan")
        return readiness

    def validate_planned_attempt(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
    ) -> AttemptManifest:
        """Validate attempt identity and scheduler task ownership against one shard."""
        _require(attempt.run_id == self.plan.run_id, "attempt run_id does not match the resolved plan")
        self._validate_plan_reference(attempt.resolved_plan)
        canonical_shard = self._get_planned_shard(attempt.shard_id)
        _require(canonical_shard == planned_shard, "planned shard is not the canonical shard for the attempt")
        expected_attempt_id = f"attempt-{attempt.attempt_ordinal:04d}"
        _require(attempt.attempt_id == expected_attempt_id, "attempt ID does not match its ordinal")
        _require(attempt.scheduler is not None, "planned attempts require scheduler array-task identity")
        _require(
            attempt.scheduler.array_task_id == planned_shard.array_task_index,
            "attempt scheduler array task does not match the planned shard",
        )
        return attempt

    def validate_finalization_chain(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
        winner: ShardWinner,
    ) -> ShardWinner:
        """Validate a complete semantic result through immutable winner publication."""
        self.validate_planned_attempt(planned_shard, attempt)
        _require(attempt.state is AttemptLifecycleState.SUCCEEDED, "only successful attempts may be finalized")
        _require(
            attempt.terminal_classification is AttemptTerminalClassification.SUCCEEDED,
            "only successfully classified attempts may be finalized",
        )
        _require(client_result.outcome is ClientOutcome.COMPLETE, "only complete client results may be finalized")
        _require(candidate.winner_eligible, "only complete candidate outputs may be finalized")

        for record_name, run_id in (
            ("client result", client_result.run_id),
            ("candidate", candidate.run_id),
            ("winner", winner.run_id),
        ):
            _require(run_id == self.plan.run_id, f"{record_name} run_id does not match the resolved plan")
        for record_name, shard_id in (
            ("client result", client_result.shard_id),
            ("candidate", candidate.shard_id),
            ("winner", winner.shard_id),
        ):
            _require(shard_id == planned_shard.shard_id, f"{record_name} shard_id does not match the planned shard")
        for record_name, attempt_id in (
            ("client result", client_result.attempt_id),
            ("candidate", candidate.attempt_id),
            ("winner", winner.attempt_id),
        ):
            _require(attempt_id == attempt.attempt_id, f"{record_name} attempt_id does not match the attempt")

        _require(
            candidate.attempt_ordinal == attempt.attempt_ordinal == winner.attempt_ordinal,
            "candidate and winner attempt ordinals must match the attempt",
        )
        _require(
            client_result.requested_records == candidate.requested_records == planned_shard.requested_records,
            "client and candidate requested records must match the planned shard",
        )
        _require(
            client_result.actual_records == candidate.actual_records == planned_shard.requested_records,
            "client and candidate actual records must complete the planned shard",
        )
        _require(
            client_result.requested_resume_mode == self.plan.invocation.authored.resume,
            "client requested resume mode does not match the resolved plan",
        )

        expected_dataset_path = self._get_expected_dataset_path(planned_shard, attempt, client_result)
        _require(
            client_result.dataset_path == expected_dataset_path, "client dataset path does not match planned intent"
        )
        _require(
            candidate.dataset_path == expected_dataset_path, "candidate dataset path does not match planned intent"
        )

        expected_manifest_path = posixpath.join(
            posixpath.dirname(planned_shard.resume_workspace.path),
            "attempts",
            attempt.attempt_id,
            "output-manifest.json",
        )
        candidate_reference = client_result.candidate_output_manifest
        _require(candidate_reference is not None, "complete client result has no candidate manifest reference")
        _require(
            candidate_reference.path == expected_manifest_path,
            "candidate manifest path does not match planned intent",
        )
        _require(
            candidate_reference.sha256 == candidate.compute_sha256(),
            "client candidate digest does not match the manifest",
        )
        _require(
            attempt.candidate_output == candidate_reference,
            "attempt candidate reference does not match client result",
        )
        _require(
            winner.candidate_manifest == candidate_reference,
            "winner candidate reference does not match client result",
        )

        _require(candidate.created_at >= attempt.created_at, "candidate creation cannot precede attempt creation")
        _require(
            client_result.completed_at >= candidate.created_at,
            "client completion cannot precede candidate creation",
        )
        _require(
            attempt.updated_at >= client_result.completed_at, "attempt completion cannot precede client completion"
        )
        _require(winner.published_at >= attempt.updated_at, "winner publication cannot precede attempt completion")
        return winner

    def _get_planned_shard(self, shard_id: ShardId) -> PlannedShard:
        planned_shard = self._shards_by_id.get(shard_id)
        if planned_shard is None:
            raise IntegrationContractError("attempt shard_id does not identify a planned shard")
        return planned_shard

    def _validate_plan_reference(self, reference: ArtifactReference) -> None:
        _require(
            reference.path == self._plan_reference.path,
            "resolved plan reference path does not match planned intent",
        )
        _require(
            reference.sha256 == self._plan_reference.sha256,
            "resolved plan reference digest does not match plan bytes",
        )

    @staticmethod
    def _get_expected_dataset_path(
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
    ) -> str:
        if client_result.effective_resume_mode == "always":
            return planned_shard.resume_workspace.path
        return posixpath.join(
            posixpath.dirname(planned_shard.resume_workspace.path),
            "attempts",
            attempt.attempt_id,
            "dataset",
        )


def validate_plan_shards(
    plan: ResolvedSlurmRunPlan,
    run: RunManifest,
    shards: tuple[ShardManifest, ...],
) -> tuple[ShardManifest, ...]:
    """Validate state shards for a one-off plan; reuse ``PlanStateValidator`` for batches."""
    return PlanStateValidator(plan).validate_plan_shards(run, shards)


def validate_initial_readiness(
    plan: ResolvedSlurmRunPlan,
    attempt: AttemptManifest,
    readiness: AttemptReadiness,
) -> AttemptReadiness:
    """Validate initial readiness for a one-off plan; reuse ``PlanStateValidator`` for batches."""
    return PlanStateValidator(plan).validate_initial_readiness(attempt, readiness)


def validate_planned_attempt(
    plan: ResolvedSlurmRunPlan,
    planned_shard: PlannedShard,
    attempt: AttemptManifest,
) -> AttemptManifest:
    """Validate one planned attempt; reuse ``PlanStateValidator`` for batches."""
    return PlanStateValidator(plan).validate_planned_attempt(planned_shard, attempt)


def validate_finalization_chain(
    plan: ResolvedSlurmRunPlan,
    planned_shard: PlannedShard,
    attempt: AttemptManifest,
    client_result: ClientResult,
    candidate: CandidateOutputManifest,
    winner: ShardWinner,
) -> ShardWinner:
    """Validate one finalization chain; reuse ``PlanStateValidator`` for batches."""
    return PlanStateValidator(plan).validate_finalization_chain(
        planned_shard,
        attempt,
        client_result,
        candidate,
        winner,
    )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise IntegrationContractError(message)


__all__ = [
    "IntegrationContractError",
    "PlanStateValidator",
    "validate_finalization_chain",
    "validate_initial_readiness",
    "validate_plan_shards",
    "validate_planned_attempt",
]
