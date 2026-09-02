# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Dataset-workspace ownership and immutable shard-winner policy."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import ValidationError

from data_designer.slurm.client import ClientResult
from data_designer.slurm.contracts import AttemptId, ShardId
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.artifacts import CandidateArtifactVerifier, VerifiedCandidateArtifacts
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import AttemptLifecycleState, AttemptManifest, RunManifest, ShardManifest
from data_designer.slurm.state.outputs import CandidateOutputManifest, ShardWinner
from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import StateContractError, validate_shard_winner


@dataclass(frozen=True, slots=True)
class _WinnerResolution:
    winner: ShardWinner
    candidate: CandidateOutputManifest
    plan: ResolvedSlurmRunPlan
    already_published: bool


class WinnerFinalizer:
    """Compose persisted state, artifact verification, and winner mutation."""

    def __init__(self, storage: StateStorage, reader: StateReader) -> None:
        self._storage = storage
        self._reader = reader
        self._artifacts = CandidateArtifactVerifier()

    @contextmanager
    def acquire_dataset_workspace(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        effective_resume_mode: Literal["never", "always"],
    ) -> Iterator[Path]:
        with self._storage.acquire_resume_lock(shard_id):
            dataset_path = self._prepare_workspace_or_normalize(shard_id, attempt_id, effective_resume_mode)
            yield dataset_path

    def finalize_winner(self, shard_id: ShardId, attempt_id: AttemptId, published_at: datetime) -> ShardWinner:
        try:
            with self._storage.acquire_resume_lock(shard_id):
                return self._finalize_with_dataset_lease(shard_id, attempt_id, published_at)
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (PlanStateContractError, StateContractError, ValidationError) as error:
            raise StateConflictError("attempt result is not eligible for winner publication") from error
        except FileNotFoundError as error:
            raise StateNotFoundError(
                f"attempt {attempt_id!r} does not contain complete finalization records"
            ) from error
        except OSError as error:
            raise SlurmStateError(f"cannot finalize winner for shard {shard_id!r}") from error

    def load_winner(self, shard_id: ShardId) -> ShardWinner:
        try:
            run, plan, shard = self._reader.load_shard_context(shard_id)
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            winner = self.load_optional_winner(run, plan, shard, attempts)
            if winner is None:
                raise StateNotFoundError(f"shard {shard_id!r} has no winner")
            return winner
        except (StateCorruptionError, StateNotFoundError):
            raise
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError(f"shard {shard_id!r} has an invalid winner chain") from error
        except (FileNotFoundError, OSError) as error:
            raise StateCorruptionError(f"shard {shard_id!r} has unreadable winner state") from error

    def require_no_winner(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempts: tuple[AttemptManifest, ...],
    ) -> None:
        if self.load_optional_winner(run, plan, shard, attempts) is not None:
            raise StateConflictError(f"shard {shard.shard_id!r} already has an immutable winner")

    def load_optional_winner(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempts: tuple[AttemptManifest, ...],
    ) -> ShardWinner | None:
        try:
            winner = self._storage.read_winner(shard.shard_id)
        except FileNotFoundError:
            return None
        except OSError as error:
            raise StateCorruptionError("persisted winner record is unsafe or unreadable") from error
        attempt = self._get_winner_attempt(attempts, winner)
        try:
            self._validate_persisted_winner(run, plan, shard, attempt, winner)
        except (PlanStateContractError, StateContractError) as error:
            raise StateCorruptionError("persisted winner chain is invalid") from error
        return winner

    def _finalize_with_dataset_lease(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        published_at: datetime,
    ) -> ShardWinner:
        resolution = self._resolve_under_state_locks(shard_id, attempt_id, published_at)
        if resolution.already_published:
            return resolution.winner
        with ExitStack() as resources:
            artifacts = self._open_candidate_artifacts(resources, resolution)
            self._validate_artifact_metadata(resolution.candidate, artifacts)
            return self._publish_verified_resolution(shard_id, attempt_id, published_at, resolution, artifacts)

    def _open_candidate_artifacts(
        self,
        resources: ExitStack,
        resolution: _WinnerResolution,
    ) -> VerifiedCandidateArtifacts:
        try:
            return resources.enter_context(self._artifacts.verify(resolution.candidate))
        except OSError as error:
            raise StateContractError("candidate output files are unavailable or invalid") from error

    def _publish_verified_resolution(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        published_at: datetime,
        expected: _WinnerResolution,
        artifacts: VerifiedCandidateArtifacts,
    ) -> ShardWinner:
        with self._storage.acquire_shard_lock(shard_id):
            current = self._resolve_winner(shard_id, attempt_id, published_at)
            if current.already_published:
                return current.winner
            if current.winner != expected.winner or current.candidate != expected.candidate:
                raise StateContractError("attempt finalization records changed during verification")
            try:
                artifacts.rebind()
            except OSError as error:
                raise StateContractError("candidate output paths changed during finalization") from error
            self._storage.publish_winner(expected.winner)
            return expected.winner

    def _resolve_under_state_locks(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        published_at: datetime,
    ) -> _WinnerResolution:
        with self._storage.acquire_shard_lock(shard_id):
            return self._resolve_winner(shard_id, attempt_id, published_at)

    def _resolve_winner(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        published_at: datetime,
    ) -> _WinnerResolution:
        run, plan, shard = self._reader.load_shard_context(shard_id)
        attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
        attempt = self._reader.get_attempt(attempts, attempt_id)
        existing = self.load_optional_winner(run, plan, shard, attempts)
        client_result, candidate = self._load_finalization_records(attempt)
        if existing is not None:
            if existing.attempt_id != attempt_id:
                raise StateConflictError(f"shard {shard_id!r} already has an immutable winner")
            return _WinnerResolution(existing, candidate, plan, True)
        winner = self._build_winner(run, shard, attempt, client_result, published_at)
        self._validate_finalization_chain(run, plan, shard, attempt, client_result, candidate, winner)
        return _WinnerResolution(winner, candidate, plan, False)

    def _prepare_dataset_workspace(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        effective_resume_mode: Literal["never", "always"],
    ) -> Path:
        with self._storage.acquire_shard_lock(shard_id):
            run, plan, shard = self._reader.load_shard_context(shard_id)
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            attempt = self._reader.get_attempt(attempts, attempt_id)
            self.require_no_winner(run, plan, shard, attempts)
            self._validate_workspace_mode(plan, attempt, effective_resume_mode)
            dataset_path = self._storage.ensure_dataset_directory(shard_id, attempt_id, effective_resume_mode)
            expected_path = (
                Path(shard.resume_workspace.path)
                if effective_resume_mode == "always"
                else self._storage.get_attempt_path(shard_id, attempt_id) / "dataset"
            )
            if dataset_path != expected_path:
                raise StateContractError("dataset workspace path does not match the resolved plan")
            return dataset_path

    def _prepare_workspace_or_normalize(
        self,
        shard_id: ShardId,
        attempt_id: AttemptId,
        effective_resume_mode: Literal["never", "always"],
    ) -> Path:
        try:
            return self._prepare_dataset_workspace(shard_id, attempt_id, effective_resume_mode)
        except (StateConflictError, StateCorruptionError, StateNotFoundError, SlurmStateError):
            raise
        except (PlanStateContractError, StateContractError) as error:
            raise StateConflictError("attempt cannot acquire the requested dataset workspace") from error
        except OSError as error:
            raise SlurmStateError(f"cannot prepare dataset workspace for attempt {attempt_id!r}") from error

    def _load_finalization_records(
        self,
        attempt: AttemptManifest,
    ) -> tuple[ClientResult, CandidateOutputManifest]:
        try:
            return self._storage.read_finalization_records(attempt.shard_id, attempt.attempt_id)
        except FileNotFoundError:
            raise
        except OSError as error:
            raise StateCorruptionError("attempt finalization records are unsafe or unreadable") from error

    def _validate_persisted_winner(
        self,
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempt: AttemptManifest,
        winner: ShardWinner,
    ) -> None:
        try:
            client_result, candidate = self._load_finalization_records(attempt)
        except FileNotFoundError as error:
            raise StateCorruptionError("persisted winner is missing its finalization records") from error
        self._validate_finalization_chain(run, plan, shard, attempt, client_result, candidate, winner)

    @staticmethod
    def _build_winner(
        run: RunManifest,
        shard: ShardManifest,
        attempt: AttemptManifest,
        client_result: ClientResult,
        published_at: datetime,
    ) -> ShardWinner:
        reference = client_result.candidate_output_manifest
        if reference is None:
            raise StateContractError("complete client result has no candidate reference")
        return ShardWinner(
            schema_version=1,
            run_id=run.run_id,
            shard_id=shard.shard_id,
            attempt_id=attempt.attempt_id,
            attempt_ordinal=attempt.attempt_ordinal,
            candidate_manifest=reference,
            published_at=published_at,
        )

    @staticmethod
    def _validate_finalization_chain(
        run: RunManifest,
        plan: ResolvedSlurmRunPlan,
        shard: ShardManifest,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
        winner: ShardWinner,
    ) -> None:
        PersistedPlanStateValidator(plan).validate_finalization_chain(
            plan.shards[shard.shard_index],
            attempt,
            client_result,
            candidate,
            winner,
        )
        validate_shard_winner(run, shard, attempt, candidate, winner)

    @staticmethod
    def _validate_artifact_metadata(
        candidate: CandidateOutputManifest,
        artifacts: VerifiedCandidateArtifacts,
    ) -> None:
        declared_counts = tuple(output_file.record_count for output_file in candidate.files)
        if artifacts.record_counts != declared_counts or sum(artifacts.record_counts) != candidate.actual_records:
            raise StateContractError("candidate Parquet row counts do not match the output manifest")
        if artifacts.dataset_schema_digest != candidate.dataset_schema_digest:
            raise StateContractError("candidate Parquet schema does not match the output manifest")

    @staticmethod
    def _validate_workspace_mode(
        plan: ResolvedSlurmRunPlan,
        attempt: AttemptManifest,
        effective_resume_mode: Literal["never", "always"],
    ) -> None:
        if attempt.state not in {
            AttemptLifecycleState.SUBMITTED,
            AttemptLifecycleState.PENDING,
            AttemptLifecycleState.RUNNING,
        }:
            raise StateContractError("dataset workspace requires an active submitted attempt")
        requested = plan.invocation.authored.resume
        if requested != "if_possible" and effective_resume_mode != requested:
            raise StateContractError("effective resume mode does not match the resolved plan")

    def _get_winner_attempt(
        self,
        attempts: tuple[AttemptManifest, ...],
        winner: ShardWinner,
    ) -> AttemptManifest:
        try:
            return self._reader.get_attempt(attempts, winner.attempt_id)
        except StateNotFoundError as error:
            raise StateCorruptionError("persisted winner references an unknown attempt") from error
