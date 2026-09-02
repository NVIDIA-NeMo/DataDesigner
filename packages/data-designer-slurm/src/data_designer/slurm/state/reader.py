# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated reads of persisted Slurm state."""

from __future__ import annotations

from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import AttemptId, Identifier, ShardId
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.errors import StateCorruptionError, StateNotFoundError
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.outputs import CandidateOutputManifest
from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError
from data_designer.slurm.state.readiness import AttemptReadiness
from data_designer.slurm.state.scheduler import SchedulerObservation
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_manifest,
    validate_attempt_set,
    validate_shard_attempt_set,
)


class StateReader:
    """Compose physical records into plan-validated state snapshots."""

    def __init__(self, storage: StateStorage, run_id: Identifier) -> None:
        self._storage = storage
        self._run_id = run_id

    def load_run(self) -> RunManifest:
        try:
            run = self._storage.read_run()
            if (
                run.run_id != self._run_id
                or run.authored_config.path != self._storage.authored_config_path.as_posix()
                or run.resolved_plan.path != self._storage.resolved_plan_path.as_posix()
            ):
                raise StateCorruptionError(f"run {self._run_id!r} manifest does not match its persisted location")
            return run
        except FileNotFoundError as error:
            raise StateNotFoundError(f"run {self._run_id!r} is not initialized") from error
        except StateCorruptionError:
            raise
        except OSError as error:
            raise StateCorruptionError(f"cannot load persisted run {self._run_id!r}") from error

    def load_authored_config(self, run: RunManifest | None = None) -> DataDesignerSlurmConfig:
        bound_run = self.load_run() if run is None else run
        try:
            authored_config = self._storage.read_authored_config()
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has no valid authored config") from error
        if (
            bound_run.authored_config.path != self._storage.authored_config_path.as_posix()
            or bound_run.authored_config.sha256 != authored_config.compute_sha256()
        ):
            raise StateCorruptionError(f"run {self._run_id!r} authored config does not match its manifest")
        return authored_config

    def load_resolved_plan(self, run: RunManifest | None = None) -> ResolvedSlurmRunPlan:
        bound_run = self.load_run() if run is None else run
        try:
            plan = self._storage.read_resolved_plan()
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has no valid resolved plan") from error
        if (
            bound_run.resolved_plan.path != self._storage.resolved_plan_path.as_posix()
            or bound_run.resolved_plan.sha256 != plan.compute_sha256()
        ):
            raise StateCorruptionError(f"run {self._run_id!r} resolved plan does not match its manifest")
        if bound_run.authored_config != plan.authored_config:
            raise StateCorruptionError(f"run {self._run_id!r} resolved plan does not bind its authored config")
        self.load_authored_config(bound_run)
        return plan

    def load_shards(
        self,
        run: RunManifest | None = None,
        plan: ResolvedSlurmRunPlan | None = None,
    ) -> tuple[ShardManifest, ...]:
        bound_run = self.load_run() if run is None else run
        bound_plan = self.load_resolved_plan(bound_run) if plan is None else plan
        try:
            shards = self._storage.read_shards(bound_run.shard_count)
            PersistedPlanStateValidator(bound_plan).validate_plan_shards(bound_run, shards)
            return shards
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has invalid persisted shards") from error
        except StateCorruptionError:
            raise
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has unreadable persisted shards") from error

    def load_context(self) -> tuple[RunManifest, ResolvedSlurmRunPlan, tuple[ShardManifest, ...]]:
        run = self.load_run()
        plan = self.load_resolved_plan(run)
        shards = self.load_shards(run, plan)
        return run, plan, shards

    def load_shard_context(self, shard_id: ShardId) -> tuple[RunManifest, ResolvedSlurmRunPlan, ShardManifest]:
        """Load and validate only the immutable context needed by one shard."""
        run = self.load_run()
        plan = self.load_resolved_plan(run)
        try:
            shard = self._storage.read_shard(shard_id)
            PersistedPlanStateValidator(plan).validate_plan_shard(run, shard)
            return run, plan, shard
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"shard {shard_id!r} is invalid") from error
        except StateCorruptionError:
            raise
        except FileNotFoundError as error:
            raise StateNotFoundError(f"shard {shard_id!r} is not persisted") from error
        except OSError as error:
            raise StateCorruptionError(f"shard {shard_id!r} is unreadable") from error

    def load_attempts(
        self,
        shard_id: ShardId,
        context: tuple[RunManifest, ResolvedSlurmRunPlan, ShardManifest] | None = None,
    ) -> tuple[AttemptManifest, ...]:
        try:
            run, plan, shard = self.load_shard_context(shard_id) if context is None else context
            if shard.shard_id != shard_id:
                raise StateContractError("shard context does not match the requested shard")
            return self.load_validated_shard_attempts(run, plan, shard)
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"shard {shard_id!r} has invalid attempts") from error
        except (StateCorruptionError, StateNotFoundError):
            raise
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"cannot load attempts for shard {shard_id!r}") from error

    def load_attempt(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptManifest:
        return self.get_attempt(self.load_attempts(shard_id), attempt_id)

    def load_readiness(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptReadiness:
        context = self.load_shard_context(shard_id)
        _, plan, _ = context
        attempt = self.get_attempt(self.load_attempts(shard_id, context), attempt_id)
        readiness = self.load_optional_readiness(plan, attempt)
        if readiness is None:
            raise StateNotFoundError(f"attempt {attempt_id!r} has no readiness snapshot")
        return readiness

    def load_optional_readiness(
        self,
        plan: ResolvedSlurmRunPlan,
        attempt: AttemptManifest,
    ) -> AttemptReadiness | None:
        """Load validated readiness when the runtime has published it."""
        try:
            readiness = self._storage.read_readiness(attempt.shard_id, attempt.attempt_id)
            PersistedPlanStateValidator(plan).validate_readiness_snapshot(attempt, readiness)
            return readiness
        except FileNotFoundError:
            return None
        except PlanStateContractError as error:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has invalid persisted readiness") from error
        except OSError as error:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has unreadable readiness") from error

    def load_optional_scheduler_observation(self, attempt: AttemptManifest) -> SchedulerObservation | None:
        """Load and identity-check the most recent scheduler observation."""
        try:
            observation = self._storage.read_scheduler_observation(attempt.shard_id, attempt.attempt_id)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has unreadable scheduler evidence") from error
        if attempt.scheduler is None or observation.scheduler != attempt.scheduler:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has mismatched scheduler evidence")
        if observation.observed_at < attempt.created_at:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has scheduler evidence before its creation")
        return observation

    def load_optional_attempt_result(
        self,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempt: AttemptManifest,
    ) -> tuple[ClientResult, CandidateOutputManifest] | None:
        """Load and validate a complete producer result pair when present."""
        try:
            client_result, candidate = self._storage.read_finalization_records(shard.shard_id, attempt.attempt_id)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has unreadable generation results") from error
        try:
            PersistedPlanStateValidator(plan).validate_attempt_result(
                plan.shards[shard.shard_index],
                attempt,
                client_result,
                candidate,
            )
        except PlanStateContractError as error:
            raise StateCorruptionError(f"attempt {attempt.attempt_id!r} has invalid generation results") from error
        return client_result, candidate

    def load_validated_attempts(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shards: tuple[ShardManifest, ...],
    ) -> dict[ShardId, tuple[AttemptManifest, ...]]:
        try:
            attempts_by_shard = {shard.shard_id: self._storage.read_attempts(shard.shard_id) for shard in shards}
            all_attempts = tuple(attempt for shard in shards for attempt in attempts_by_shard[shard.shard_id])
            for shard in shards:
                for attempt in attempts_by_shard[shard.shard_id]:
                    self.validate_attempt_against_plan(run, plan, shard, attempt)
            validate_attempt_set(run, shards, all_attempts)
            return attempts_by_shard
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"run {self._run_id!r} has invalid persisted attempts") from error

    def load_validated_shard_attempts(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
    ) -> tuple[AttemptManifest, ...]:
        """Read and plan-validate attempts for one shard only."""
        try:
            attempts = self._storage.read_attempts(shard.shard_id)
            for attempt in attempts:
                self.validate_attempt_against_plan(run, plan, shard, attempt)
            validate_shard_attempt_set(run, shard, attempts)
            return attempts
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"shard {shard.shard_id!r} has invalid persisted attempts") from error

    @staticmethod
    def validate_attempt_against_plan(
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempt: AttemptManifest,
    ) -> None:
        validate_attempt_manifest(run, shard, attempt)
        planned_shard = plan.shards[shard.shard_index]
        if planned_shard.shard_id != attempt.shard_id:
            raise StateContractError("attempt shard does not match resolved plan order")
        if attempt.attempt_id != f"attempt-{attempt.attempt_ordinal:04d}":
            raise StateContractError("attempt ID does not match its ordinal")
        if attempt.scheduler is not None and attempt.scheduler.array_task_id != planned_shard.array_task_index:
            raise StateContractError("attempt scheduler task does not match the resolved plan shard")
        if attempt.state is not AttemptLifecycleState.CREATED:
            PersistedPlanStateValidator(plan).validate_planned_attempt(planned_shard, attempt)

    @staticmethod
    def get_shard(shards: tuple[ShardManifest, ...], shard_id: ShardId) -> ShardManifest:
        try:
            return next(shard for shard in shards if shard.shard_id == shard_id)
        except StopIteration:
            raise StateNotFoundError(f"shard {shard_id!r} is not persisted") from None

    @staticmethod
    def get_attempt(attempts: tuple[AttemptManifest, ...], attempt_id: AttemptId) -> AttemptManifest:
        try:
            return next(attempt for attempt in attempts if attempt.attempt_id == attempt_id)
        except StopIteration:
            raise StateNotFoundError(f"attempt {attempt_id!r} is not persisted") from None
