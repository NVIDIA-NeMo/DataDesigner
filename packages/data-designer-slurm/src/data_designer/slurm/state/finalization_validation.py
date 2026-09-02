# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Semantic validation for complete Slurm candidate chains."""

from __future__ import annotations

import posixpath

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.planning import PlannedShard, ResolvedSlurmRunPlan
from data_designer.slurm.state.execution import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
)
from data_designer.slurm.state.outputs import CANDIDATE_OUTPUT_FORMAT, CandidateOutputManifest, ShardWinner


class FinalizationContractError(ValueError):
    """Raised when a candidate chain disagrees with its resolved plan."""


class FinalizationChainValidator:
    """Validate one successful attempt through immutable winner publication."""

    def __init__(self, plan: ResolvedSlurmRunPlan) -> None:
        self._plan = plan

    def validate(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
        winner: ShardWinner,
    ) -> ShardWinner:
        """Validate the identities, outputs, references, and chronology of one chain."""
        self._validate_winner_outcome(attempt)
        self.validate_attempt_result(planned_shard, attempt, client_result, candidate)
        self._validate_winner_identity(attempt, winner)
        self._validate_winner_reference(attempt, client_result, winner)
        self._validate_winner_timestamp(attempt, client_result, winner)
        return winner

    def validate_attempt_result(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        """Validate producer-owned records before immutable publication."""
        self._validate_result_outcomes(attempt, client_result, candidate)
        self._validate_result_identities(planned_shard, attempt, client_result, candidate)
        self._validate_output_contract(planned_shard, client_result, candidate)
        self._validate_dataset_paths(planned_shard, attempt, client_result, candidate)
        self._validate_result_reference(planned_shard, attempt, client_result, candidate)
        self._validate_result_timestamps(attempt, client_result, candidate)

    @staticmethod
    def _validate_result_outcomes(
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        _require(
            attempt.state in {AttemptLifecycleState.RUNNING, AttemptLifecycleState.SUCCEEDED},
            "attempt results require a running or successful attempt",
        )
        _require(client_result.outcome is ClientOutcome.COMPLETE, "only complete client results may be finalized")
        _require(candidate.winner_eligible, "only complete candidate outputs may be finalized")

    @staticmethod
    def _validate_winner_outcome(attempt: AttemptManifest) -> None:
        _require(attempt.state is AttemptLifecycleState.SUCCEEDED, "only successful attempts may be finalized")
        _require(
            attempt.terminal_classification is AttemptTerminalClassification.SUCCEEDED,
            "only successfully classified attempts may be finalized",
        )

    def _validate_result_identities(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        for record_name, run_id in (
            ("client result", client_result.run_id),
            ("candidate", candidate.run_id),
        ):
            _require(run_id == self._plan.run_id, f"{record_name} run_id does not match the resolved plan")
        for record_name, shard_id in (
            ("client result", client_result.shard_id),
            ("candidate", candidate.shard_id),
        ):
            _require(shard_id == planned_shard.shard_id, f"{record_name} shard_id does not match the planned shard")
        for record_name, attempt_id in (
            ("client result", client_result.attempt_id),
            ("candidate", candidate.attempt_id),
        ):
            _require(attempt_id == attempt.attempt_id, f"{record_name} attempt_id does not match the attempt")
        _require(
            candidate.attempt_ordinal == attempt.attempt_ordinal,
            "candidate and attempt ordinals must match",
        )

    def _validate_winner_identity(self, attempt: AttemptManifest, winner: ShardWinner) -> None:
        _require(winner.run_id == self._plan.run_id, "winner run_id does not match the resolved plan")
        _require(winner.shard_id == attempt.shard_id, "winner shard_id does not match the attempt")
        _require(winner.attempt_id == attempt.attempt_id, "winner attempt_id does not match the attempt")
        _require(winner.attempt_ordinal == attempt.attempt_ordinal, "winner and attempt ordinals must match")

    def _validate_output_contract(
        self,
        planned_shard: PlannedShard,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        _require(
            client_result.requested_records == candidate.requested_records == planned_shard.requested_records,
            "client and candidate requested records must match the planned shard",
        )
        _require(
            client_result.actual_records == candidate.actual_records == planned_shard.requested_records,
            "client and candidate actual records must complete the planned shard",
        )
        expected_suffix = f".{CANDIDATE_OUTPUT_FORMAT}"
        _require(
            all(output_file.relative_path.endswith(expected_suffix) for output_file in candidate.files),
            "candidate output file extensions must use the attempt-local Parquet format",
        )
        _require(
            all(output_file.byte_size > 0 for output_file in candidate.files if output_file.record_count > 0),
            "non-empty candidate output files must contain bytes",
        )
        _require(
            candidate.provenance_digest == self._plan.compute_sha256(),
            "candidate provenance digest does not match the resolved plan",
        )
        _require(
            client_result.requested_resume_mode == self._plan.invocation.authored.resume,
            "client requested resume mode does not match the resolved plan",
        )

    def _validate_dataset_paths(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        expected_path = self._get_expected_dataset_path(planned_shard, attempt, client_result)
        _require(client_result.dataset_path == expected_path, "client dataset path does not match planned intent")
        _require(candidate.dataset_path == expected_path, "candidate dataset path does not match planned intent")

    @staticmethod
    def _validate_result_reference(
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        expected_path = posixpath.join(
            posixpath.dirname(planned_shard.resume_workspace.path),
            "attempts",
            attempt.attempt_id,
            "output-manifest.json",
        )
        reference = client_result.candidate_output_manifest
        _require(reference is not None, "complete client result has no candidate manifest reference")
        _require(reference.path == expected_path, "candidate manifest path does not match planned intent")
        _require(reference.sha256 == candidate.compute_sha256(), "client candidate digest does not match the manifest")

    @staticmethod
    def _validate_winner_reference(
        attempt: AttemptManifest,
        client_result: ClientResult,
        winner: ShardWinner,
    ) -> None:
        reference = client_result.candidate_output_manifest
        _require(reference is not None, "complete client result has no candidate manifest reference")
        _require(attempt.candidate_output == reference, "attempt candidate reference does not match client result")
        _require(winner.candidate_manifest == reference, "winner candidate reference does not match client result")

    @staticmethod
    def _validate_result_timestamps(
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        _require(candidate.created_at >= attempt.created_at, "candidate creation cannot precede attempt creation")
        _require(
            client_result.completed_at >= candidate.created_at, "client completion cannot precede candidate creation"
        )

    @staticmethod
    def _validate_winner_timestamp(
        attempt: AttemptManifest,
        client_result: ClientResult,
        winner: ShardWinner,
    ) -> None:
        _require(
            attempt.updated_at >= client_result.completed_at,
            "attempt completion cannot precede client completion",
        )
        _require(
            winner.published_at >= attempt.updated_at,
            "winner publication cannot precede attempt completion",
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


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise FinalizationContractError(message)


__all__ = ["FinalizationChainValidator", "FinalizationContractError"]
