# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validated publication and attempt-binding boundary for producer-owned results."""

from __future__ import annotations

from data_designer.slurm.client import ClientResult
from data_designer.slurm.state.errors import (
    SlurmStateError,
    StateConflictError,
    StateCorruptionError,
    StateNotFoundError,
)
from data_designer.slurm.state.execution import AttemptManifest
from data_designer.slurm.state.outputs import CandidateOutputManifest
from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import StateContractError, validate_attempt_transition


class AttemptResultPublisher:
    """Publish one result pair and bind its candidate reference as the commit marker."""

    def __init__(self, storage: StateStorage, reader: StateReader) -> None:
        self._storage = storage
        self._reader = reader

    def publish(
        self,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> tuple[ClientResult, CandidateOutputManifest]:
        """Publish a result pair and bind the attempt under its shard lock."""
        self._validate_record_location(client_result, candidate)
        try:
            with self._storage.acquire_shard_lock(candidate.shard_id):
                run, plan, shard = self._reader.load_shard_context(candidate.shard_id)
                attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
                attempt = self._reader.get_attempt(attempts, candidate.attempt_id)
                PersistedPlanStateValidator(plan).validate_attempt_result(
                    plan.shards[shard.shard_index],
                    attempt,
                    client_result,
                    candidate,
                )
                bound_attempt = self._bind_candidate_reference(attempt, client_result)
                self._storage.publish_finalization_records(client_result, candidate)
                if bound_attempt != attempt:
                    self._storage.replace_attempt(bound_attempt)
                else:
                    self._storage.sync_attempt_directory(attempt.shard_id, attempt.attempt_id)
            return client_result, candidate
        except (StateConflictError, StateCorruptionError, StateNotFoundError):
            raise
        except (FileExistsError, PlanStateContractError, StateContractError) as error:
            raise StateConflictError("attempt result does not match persisted run intent") from error
        except OSError as error:
            raise SlurmStateError(f"cannot publish result for attempt {candidate.attempt_id!r}") from error

    def _validate_record_location(
        self,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        if not isinstance(client_result, ClientResult) or not isinstance(candidate, CandidateOutputManifest):
            raise StateConflictError("attempt result records have invalid types")
        if client_result.run_id != self._storage.run_id or candidate.run_id != self._storage.run_id:
            raise StateConflictError("attempt result run identity does not match the state writer")
        if client_result.shard_id != candidate.shard_id or client_result.attempt_id != candidate.attempt_id:
            raise StateConflictError("attempt result records do not identify the same attempt")

    @staticmethod
    def _bind_candidate_reference(attempt: AttemptManifest, client_result: ClientResult) -> AttemptManifest:
        reference = client_result.candidate_output_manifest
        if reference is None:
            raise StateContractError("complete client result has no candidate manifest reference")
        if attempt.candidate_output == reference:
            return attempt
        bound_attempt = attempt.model_copy(update={"candidate_output": reference})
        validate_attempt_transition(attempt, bound_attempt)
        return bound_attempt


__all__ = ["AttemptResultPublisher"]
