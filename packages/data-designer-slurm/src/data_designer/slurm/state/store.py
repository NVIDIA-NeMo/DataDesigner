# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated state transitions for one persisted Slurm run."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from pydantic import TypeAdapter, ValidationError

from data_designer.slurm.client import ClientResult
from data_designer.slurm.config import DataDesignerSlurmConfig
from data_designer.slurm.contracts import AttemptId, Identifier, ShardId, validate_absolute_path
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.finalization import WinnerFinalizer
from data_designer.slurm.state.outputs import CandidateOutputManifest, ShardWinner
from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.readiness import AttemptReadiness
from data_designer.slurm.state.reconciliation import validate_readiness_transition
from data_designer.slurm.state.results import AttemptResultPublisher
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import (
    StateContractError,
    validate_attempt_transition,
    validate_shard_attempt_set,
)

_IDENTIFIER_ADAPTER = TypeAdapter(Identifier)
_SHARD_ID_ADAPTER = TypeAdapter(ShardId)
_ATTEMPT_ID_ADAPTER = TypeAdapter(AttemptId)


class SlurmStateWriter:
    """Persist and reload one run's immutable and revisioned state.

    The writer owns state-machine and plan validation while its composed
    storage collaborator owns descriptor-bound paths, locks, serialization,
    and publication. Attempt and readiness mutations lock and validate only
    the target shard; full-run validation remains available for status and
    audit snapshots.

    Args:
        workspace_root: Selected compute-visible workspace root.
        run_id: Stable application-owned run identity.
    """

    def __init__(self, workspace_root: str | Path, run_id: Identifier) -> None:
        try:
            normalized_root = validate_absolute_path(Path(workspace_root).as_posix())
            normalized_run_id = _IDENTIFIER_ADAPTER.validate_python(run_id, strict=True)
        except (ValidationError, ValueError) as error:
            raise SlurmStateError("invalid persisted run location") from error
        self._storage = StateStorage(Path(normalized_root), normalized_run_id)
        self._reader = StateReader(self._storage, normalized_run_id)
        self._results = AttemptResultPublisher(self._storage, self._reader)
        self._finalizer = WinnerFinalizer(self._storage, self._reader)
        self._run_id = normalized_run_id

    @property
    def run_root(self) -> Path:
        """Return the workspace-derived root for this run."""
        return self._storage.run_root

    def initialize_run(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> RunManifest:
        """Convergently publish immutable run intent, plan, and shards."""
        self._validate_initial_state(authored_config, resolved_plan, run, shards)
        try:
            self._storage.ensure_storage()
            with self._storage.acquire_run_lock():
                self._storage.publish_initial_state(authored_config, resolved_plan, run, shards)
            return run
        except StateConflictError:
            raise
        except FileExistsError as error:
            raise StateConflictError(f"run {self._run_id!r} already contains different immutable state") from error
        except OSError as error:
            raise SlurmStateError(f"cannot initialize persisted run {self._run_id!r}") from error

    def load_run(self) -> RunManifest:
        """Load the committed immutable run manifest."""
        return self._reader.load_run()

    def load_authored_config(self) -> DataDesignerSlurmConfig:
        """Load and digest-verify the run's immutable authored config."""
        return self._reader.load_authored_config()

    def load_resolved_plan(self) -> ResolvedSlurmRunPlan:
        """Load and digest-verify the run's immutable resolved plan."""
        return self._reader.load_resolved_plan()

    def load_shards(self) -> tuple[ShardManifest, ...]:
        """Load and validate the complete ordered shard set."""
        return self._reader.load_shards()

    def create_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        """Publish the next monotonically numbered attempt for one shard."""
        self._validate_attempt_location(attempt)
        try:
            return self._create_attempt_with_locks(attempt)
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (PlanStateContractError, StateContractError) as error:
            raise StateConflictError("attempt does not match persisted run intent") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot create attempt {attempt.attempt_id!r}") from error

    def update_attempt(self, attempt: AttemptManifest) -> AttemptManifest:
        """Atomically replace one attempt after validating its monotonic transition."""
        self._validate_attempt_location(attempt)
        try:
            return self._update_attempt_with_locks(attempt)
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (PlanStateContractError, StateContractError) as error:
            raise StateConflictError("attempt update is not a valid monotonic transition") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot update attempt {attempt.attempt_id!r}") from error

    def load_attempts(self, shard_id: ShardId) -> tuple[AttemptManifest, ...]:
        """Load one shard's attempts in ordinal order."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        return self._reader.load_attempts(normalized_shard_id)

    def load_attempt(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptManifest:
        """Load one persisted attempt."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        return self._reader.load_attempt(normalized_shard_id, normalized_attempt_id)

    def write_readiness(self, readiness: AttemptReadiness) -> AttemptReadiness:
        """Create or atomically replace one validated readiness snapshot."""
        self._validate_readiness_location(readiness)
        try:
            return self._write_readiness_with_lock(readiness)
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (PlanStateContractError, StateContractError) as error:
            raise StateConflictError("readiness update is not a valid monotonic transition") from error
        except (FileNotFoundError, OSError) as error:
            raise SlurmStateError(f"cannot persist readiness for attempt {readiness.attempt_id!r}") from error

    def load_readiness(self, shard_id: ShardId, attempt_id: AttemptId) -> AttemptReadiness:
        """Load one attempt's latest readiness snapshot."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        return self._reader.load_readiness(normalized_shard_id, normalized_attempt_id)

    def publish_attempt_result(
        self,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> tuple[ClientResult, CandidateOutputManifest]:
        """Publish one result pair and atomically bind its candidate to the attempt."""
        return self._results.publish(client_result, candidate)

    @contextmanager
    def acquire_dataset_workspace(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        effective_resume_mode: Literal["never", "always"],
    ) -> Iterator[Path]:
        """Yield one validated dataset path while holding its shard lease."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        if type(effective_resume_mode) is not str or effective_resume_mode not in {"never", "always"}:
            raise StateConflictError("effective resume mode must be 'never' or 'always'")
        with self._finalizer.acquire_dataset_workspace(
            normalized_shard_id,
            normalized_attempt_id,
            effective_resume_mode,
        ) as dataset_path:
            yield dataset_path

    def finalize_winner(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        *,
        published_at: datetime,
    ) -> ShardWinner:
        """Validate attempt-local artifacts and publish one immutable winner."""
        normalized_shard_id = self._validate_shard_id(shard_id)
        normalized_attempt_id = self._validate_attempt_id(attempt_id)
        if (
            not isinstance(published_at, datetime)
            or published_at.tzinfo is None
            or published_at.utcoffset() != timedelta(0)
        ):
            raise StateConflictError("winner publication timestamp must be timezone-aware UTC")
        return self._finalizer.finalize_winner(normalized_shard_id, normalized_attempt_id, published_at)

    def load_winner(self, shard_id: ShardId) -> ShardWinner:
        """Load and validate one shard's immutable winner chain."""
        return self._finalizer.load_winner(self._validate_shard_id(shard_id))

    def _create_attempt_with_locks(self, attempt: AttemptManifest) -> AttemptManifest:
        with self._storage.acquire_resume_and_shard_locks(attempt.shard_id):
            run, plan, shard = self._reader.load_shard_context(attempt.shard_id)
            shard_attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            existing = next((item for item in shard_attempts if item.attempt_id == attempt.attempt_id), None)
            if existing is not None:
                if existing != attempt:
                    raise StateConflictError(f"attempt {attempt.attempt_id!r} already contains different state")
                self._storage.sync_attempt_directory(attempt.shard_id, attempt.attempt_id)
                return existing
            self._finalizer.require_no_winner(run, plan, shard, shard_attempts)
            expected_ordinal = len(shard_attempts) + 1
            if attempt.attempt_ordinal != expected_ordinal or attempt.attempt_id != f"attempt-{expected_ordinal:04d}":
                raise StateConflictError("attempt identity does not match the next shard ordinal")
            self._reader.validate_attempt_against_plan(run, plan, shard, attempt)
            validate_shard_attempt_set(run, shard, shard_attempts + (attempt,))
            self._storage.publish_attempt(attempt)
            return attempt

    def _update_attempt_with_locks(self, attempt: AttemptManifest) -> AttemptManifest:
        with self._storage.acquire_shard_lock(attempt.shard_id):
            run, plan, shard = self._reader.load_shard_context(attempt.shard_id)
            shard_attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            previous = self._reader.get_attempt(shard_attempts, attempt.attempt_id)
            if previous == attempt:
                self._storage.sync_attempt_directory(attempt.shard_id, attempt.attempt_id)
                return previous
            validate_attempt_transition(previous, attempt)
            if previous.candidate_output is None and attempt.candidate_output is not None:
                raise StateContractError("candidate output must be bound by result publication")
            self._reader.validate_attempt_against_plan(run, plan, shard, attempt)
            updated_attempts = tuple(
                attempt if item.shard_id == attempt.shard_id and item.attempt_id == attempt.attempt_id else item
                for item in shard_attempts
            )
            validate_shard_attempt_set(run, shard, updated_attempts)
            self._storage.replace_attempt(attempt)
            return attempt

    def _write_readiness_with_lock(self, readiness: AttemptReadiness) -> AttemptReadiness:
        with self._storage.acquire_shard_lock(readiness.shard_id):
            run, plan, shard = self._reader.load_shard_context(readiness.shard_id)
            shard_attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            attempt = self._reader.get_attempt(shard_attempts, readiness.attempt_id)
            previous = self._load_optional_readiness(readiness)
            if previous is None:
                PersistedPlanStateValidator(plan).validate_initial_readiness(attempt, readiness)
                self._storage.publish_readiness(readiness)
                return readiness
            self._validate_persisted_readiness(plan, attempt, previous)
            if previous == readiness:
                self._storage.sync_attempt_directory(readiness.shard_id, readiness.attempt_id)
                return previous
            validate_readiness_transition(previous, readiness)
            self._storage.replace_readiness(readiness)
            return readiness

    def _load_optional_readiness(self, readiness: AttemptReadiness) -> AttemptReadiness | None:
        try:
            return self._storage.read_readiness(readiness.shard_id, readiness.attempt_id)
        except FileNotFoundError:
            return None

    @staticmethod
    def _validate_persisted_readiness(
        plan: ResolvedSlurmRunPlan,
        attempt: AttemptManifest,
        readiness: AttemptReadiness,
    ) -> None:
        try:
            PersistedPlanStateValidator(plan).validate_readiness_snapshot(attempt, readiness)
        except PlanStateContractError as error:
            raise StateCorruptionError(f"attempt {readiness.attempt_id!r} has invalid persisted readiness") from error

    def _validate_initial_state(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> None:
        try:
            self._validate_initial_record_types(authored_config, resolved_plan, run, shards)
            self._validate_initial_bindings(authored_config, resolved_plan, run)
            PersistedPlanStateValidator(resolved_plan).validate_plan_shards(run, shards)
        except (PlanStateContractError, StateContractError) as error:
            raise StateConflictError("run initialization does not match resolved plan intent") from error

    @staticmethod
    def _validate_initial_record_types(
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
        shards: tuple[ShardManifest, ...],
    ) -> None:
        if not isinstance(authored_config, DataDesignerSlurmConfig):
            raise StateContractError("authored config has an invalid type")
        if not isinstance(resolved_plan, ResolvedSlurmRunPlan):
            raise StateContractError("resolved plan has an invalid type")
        if not isinstance(run, RunManifest):
            raise StateContractError("run manifest has an invalid type")
        if not isinstance(shards, tuple) or any(not isinstance(shard, ShardManifest) for shard in shards):
            raise StateContractError("shard manifests have an invalid type")

    def _validate_initial_bindings(
        self,
        authored_config: DataDesignerSlurmConfig,
        resolved_plan: ResolvedSlurmRunPlan,
        run: RunManifest,
    ) -> None:
        if run.run_id != self._run_id or resolved_plan.run_id != self._run_id:
            raise StateContractError("run identity does not match the state writer")
        if resolved_plan.selected_profile.profile.workspace_root != self._storage.workspace_root.as_posix():
            raise StateContractError("resolved plan workspace does not match the state writer")
        if resolved_plan.authored_config.sha256 != authored_config.compute_sha256():
            raise StateContractError("authored config digest does not match the resolved plan")
        if run.authored_config != resolved_plan.authored_config:
            raise StateContractError("run authored config does not match the resolved plan")
        if run.authored_config.path != self._storage.authored_config_path.as_posix():
            raise StateContractError("run authored config reference does not match its persisted location")
        if (
            run.resolved_plan.path != self._storage.resolved_plan_path.as_posix()
            or run.resolved_plan.sha256 != resolved_plan.compute_sha256()
        ):
            raise StateContractError("run resolved plan reference does not match persisted plan bytes")

    def _validate_attempt_location(self, attempt: AttemptManifest) -> None:
        if not isinstance(attempt, AttemptManifest) or attempt.run_id != self._run_id:
            raise StateConflictError("attempt identity does not match the state writer")

    def _validate_readiness_location(self, readiness: AttemptReadiness) -> None:
        if not isinstance(readiness, AttemptReadiness) or readiness.run_id != self._run_id:
            raise StateConflictError("readiness identity does not match the state writer")

    @staticmethod
    def _validate_shard_id(shard_id: ShardId) -> ShardId:
        try:
            return _SHARD_ID_ADAPTER.validate_python(shard_id, strict=True)
        except ValidationError as error:
            raise SlurmStateError("invalid shard identity") from error

    @staticmethod
    def _validate_attempt_id(attempt_id: AttemptId) -> AttemptId:
        try:
            return _ATTEMPT_ID_ADAPTER.validate_python(attempt_id, strict=True)
        except ValidationError as error:
            raise SlurmStateError("invalid attempt identity") from error
