# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validation that binds persisted run state to one resolved Slurm plan."""

from __future__ import annotations

import posixpath
from collections.abc import Mapping
from types import MappingProxyType

from data_designer.slurm.contracts import ArtifactReference, ShardId
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.readiness import AttemptReadiness, ReadinessState
from data_designer.slurm.state.validation import StateContractError, validate_shard_manifest, validate_shard_set


class PlanStateContractError(ValueError):
    """Raised when persisted state disagrees with its resolved plan."""


class PersistedPlanStateValidator:
    """Validate persisted records against one resolved plan."""

    def __init__(self, plan: ResolvedSlurmRunPlan) -> None:
        self._plan: ResolvedSlurmRunPlan = plan
        run_root = posixpath.dirname(plan.authored_config.path)
        self._plan_reference: ArtifactReference = ArtifactReference(
            path=posixpath.join(run_root, "resolved-plan.json"),
            sha256=plan.compute_sha256(),
        )
        self._shards_by_id: Mapping[ShardId, PlannedShard] = MappingProxyType(
            {planned_shard.shard_id: planned_shard for planned_shard in plan.shards}
        )

    @property
    def plan(self) -> ResolvedSlurmRunPlan:
        """Return the resolved plan used for validation."""
        return self._plan

    def validate_plan_shards(
        self,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> tuple[ShardManifest, ...]:
        """Validate the complete ordered state shard set against planned intent."""
        self._validate_run(run)
        try:
            validate_shard_set(run, shards)
        except StateContractError as error:
            raise PlanStateContractError(str(error)) from error
        _require(len(shards) == len(self.plan.shards), "state shards must exactly match the planned shard count")
        for planned, persisted in zip(self.plan.shards, shards, strict=True):
            _require(planned.shard_id == persisted.shard_id, "state shard identity does not match planned order")
            self.validate_plan_shard(run, persisted)
        return shards

    def validate_plan_shard(
        self,
        run: RunManifest,
        persisted: ShardManifest,
    ) -> ShardManifest:
        """Validate one persisted shard against its canonical planned shard."""
        self._validate_run(run)
        try:
            validate_shard_manifest(run, persisted)
        except StateContractError as error:
            raise PlanStateContractError(str(error)) from error
        planned = self._get_planned_shard(persisted.shard_id)
        _require(persisted.shard_index == planned.shard_index, "state shard index does not match planned order")
        _require(persisted.record_range == planned.record_range, "state shard record range does not match the plan")
        _require(
            persisted.input_partition == planned.input_partition, "state shard input partition does not match the plan"
        )
        _require(
            persisted.resume_workspace == planned.resume_workspace,
            "state shard resume workspace does not match the plan",
        )
        return persisted

    def validate_initial_readiness(
        self,
        attempt: AttemptManifest,
        readiness: AttemptReadiness,
    ) -> AttemptReadiness:
        """Anchor the first readiness snapshot to a fully validated planned attempt."""
        self.validate_readiness_snapshot(attempt, readiness)
        _require(readiness.revision == 1, "initial readiness must use revision 1")
        _require(readiness.state is ReadinessState.PENDING, "initial readiness must be pending")
        return readiness

    def validate_readiness_snapshot(
        self,
        attempt: AttemptManifest,
        readiness: AttemptReadiness,
    ) -> AttemptReadiness:
        """Bind any persisted readiness revision to its planned attempt and deployments."""
        planned_shard = self._get_planned_shard(attempt.shard_id)
        self.validate_planned_attempt(planned_shard, attempt)
        _require(readiness.run_id == attempt.run_id, "readiness run_id does not match the attempt")
        _require(readiness.shard_id == attempt.shard_id, "readiness shard_id does not match the attempt")
        _require(readiness.attempt_id == attempt.attempt_id, "readiness attempt_id does not match the attempt")
        _require(readiness.updated_at >= attempt.created_at, "readiness cannot precede attempt creation")
        expected = tuple(
            (deployment.deployment_id, deployment.authored.model_alias, deployment.topology.replica_count)
            for deployment in self.plan.deployments
        )
        actual = tuple(
            (deployment.deployment_id, deployment.model_alias, deployment.expected_backends)
            for deployment in readiness.deployments
        )
        _require(actual == expected, "readiness deployments do not match the resolved plan")
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
        _require(
            attempt.attempt_id == f"attempt-{attempt.attempt_ordinal:04d}", "attempt ID does not match its ordinal"
        )
        _require(
            attempt.state is not AttemptLifecycleState.CREATED,
            "planned attempts must be submitted before scheduler task validation",
        )
        _require(attempt.scheduler is not None, "planned attempts require scheduler array-task identity")
        _require(
            attempt.scheduler.array_task_id == planned_shard.array_task_index,
            "attempt scheduler array task does not match the planned shard",
        )
        return attempt

    def _get_planned_shard(self, shard_id: ShardId) -> PlannedShard:
        planned_shard = self._shards_by_id.get(shard_id)
        if planned_shard is None:
            raise PlanStateContractError("attempt shard_id does not identify a planned shard")
        return planned_shard

    def _validate_run(self, run: RunManifest) -> None:
        _require(run.run_id == self.plan.run_id, "run manifest identity does not match the resolved plan")
        _require(
            run.authored_config == self.plan.authored_config, "run authored config does not match the resolved plan"
        )
        self._validate_plan_reference(run.resolved_plan)
        _require(run.shard_count == len(self.plan.shards), "run shard count does not match the resolved plan")

    def _validate_plan_reference(self, reference: ArtifactReference) -> None:
        _require(
            reference.path == self._plan_reference.path, "resolved plan reference path does not match planned intent"
        )
        _require(
            reference.sha256 == self._plan_reference.sha256,
            "resolved plan reference digest does not match plan bytes",
        )


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PlanStateContractError(message)


__all__ = ["PersistedPlanStateValidator", "PlanStateContractError"]
