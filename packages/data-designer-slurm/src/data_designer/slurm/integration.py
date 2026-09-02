# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pure validation across Slurm execution-plan and runtime-state contracts."""

from __future__ import annotations

import posixpath

from data_designer.slurm.client import ClientOutcome, ClientResult
from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import PlannedShard
from data_designer.slurm.state.execution import (
    AttemptLifecycleState,
    AttemptManifest,
    AttemptTerminalClassification,
)
from data_designer.slurm.state.outputs import CandidateOutputManifest, ShardWinner
from data_designer.slurm.state.plan_validation import PersistedPlanStateValidator, PlanStateContractError

IntegrationContractError = PlanStateContractError


class PlanStateValidator(PersistedPlanStateValidator):
    """Validate records against one resolved plan with reusable derived state.

    This is an in-process validation service, not a persisted contract record.
    """

    def validate_finalization_chain(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
        winner: ShardWinner,
    ) -> ShardWinner:
        """Validate a complete semantic result through immutable winner publication."""
        self.validate_client_candidate(planned_shard, attempt, client_result, candidate)
        _require(attempt.state is AttemptLifecycleState.SUCCEEDED, "only successful attempts may be finalized")
        _require(
            attempt.terminal_classification is AttemptTerminalClassification.SUCCEEDED,
            "only successfully classified attempts may be finalized",
        )
        self._validate_winner(planned_shard, attempt, client_result, winner)
        _require(
            attempt.updated_at >= client_result.completed_at, "attempt completion cannot precede client completion"
        )
        _require(winner.published_at >= attempt.updated_at, "winner publication cannot precede attempt completion")
        return winner

    def validate_client_candidate(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> CandidateOutputManifest:
        """Validate a complete client result and candidate before terminal publication."""
        self.validate_planned_attempt(planned_shard, attempt)
        _require(client_result.outcome is ClientOutcome.COMPLETE, "only complete client results may be finalized")
        _require(candidate.winner_eligible, "only complete candidate outputs may be finalized")
        self._validate_candidate_identities(planned_shard, attempt, client_result, candidate)
        self._validate_candidate_counts(planned_shard, client_result, candidate)
        self._validate_candidate_location(planned_shard, attempt, client_result, candidate)
        _require(candidate.created_at >= attempt.created_at, "candidate creation cannot precede attempt creation")
        _require(
            client_result.completed_at >= candidate.created_at,
            "client completion cannot precede candidate creation",
        )
        return candidate

    def _validate_candidate_identities(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
        for record_name, run_id in (("client result", client_result.run_id), ("candidate", candidate.run_id)):
            _require(run_id == self.plan.run_id, f"{record_name} run_id does not match the resolved plan")
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
            candidate.attempt_ordinal == attempt.attempt_ordinal, "candidate attempt ordinal does not match attempt"
        )

    @staticmethod
    def _validate_candidate_counts(
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
        expected_suffix = f".{self.plan.output.format}"
        _require(
            all(output_file.relative_path.endswith(expected_suffix) for output_file in candidate.files),
            "candidate output file extensions do not match the resolved output format",
        )
        _require(
            all(output_file.byte_size > 0 for output_file in candidate.files if output_file.record_count > 0),
            "non-empty candidate output files must contain bytes",
        )

    def _validate_candidate_location(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        candidate: CandidateOutputManifest,
    ) -> None:
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
        candidate_reference = self._get_candidate_reference(client_result)
        expected_manifest_path = posixpath.join(
            posixpath.dirname(planned_shard.resume_workspace.path),
            "attempts",
            attempt.attempt_id,
            "output-manifest.json",
        )
        _require(
            candidate_reference.path == expected_manifest_path, "candidate manifest path does not match planned intent"
        )
        _require(
            candidate_reference.sha256 == candidate.compute_sha256(),
            "client candidate digest does not match the manifest",
        )

    def _validate_winner(
        self,
        planned_shard: PlannedShard,
        attempt: AttemptManifest,
        client_result: ClientResult,
        winner: ShardWinner,
    ) -> None:
        candidate_reference = self._get_candidate_reference(client_result)
        _require(winner.run_id == self.plan.run_id, "winner run_id does not match the resolved plan")
        _require(winner.shard_id == planned_shard.shard_id, "winner shard_id does not match the planned shard")
        _require(winner.attempt_id == attempt.attempt_id, "winner attempt_id does not match the attempt")
        _require(winner.attempt_ordinal == attempt.attempt_ordinal, "winner attempt ordinal does not match the attempt")
        _require(
            attempt.candidate_output == candidate_reference,
            "attempt candidate reference does not match client result",
        )
        _require(
            winner.candidate_manifest == candidate_reference,
            "winner candidate reference does not match client result",
        )

    @staticmethod
    def _get_candidate_reference(client_result: ClientResult) -> ArtifactReference:
        candidate_reference = client_result.candidate_output_manifest
        _require(candidate_reference is not None, "complete client result has no candidate manifest reference")
        return candidate_reference

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
        raise IntegrationContractError(message)


__all__ = [
    "IntegrationContractError",
    "PlanStateValidator",
]
