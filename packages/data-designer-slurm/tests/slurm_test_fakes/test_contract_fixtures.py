# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.slurm.client import ClientResult
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state import (
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    RunManifest,
    ShardManifest,
    ShardWinner,
)


def test_shared_contract_fixtures_load_canonical_records(
    single_node_plan: ResolvedSlurmRunPlan,
    multi_node_plan: ResolvedSlurmRunPlan,
    client_result: ClientResult,
    run_manifest: RunManifest,
    shard_manifests: tuple[ShardManifest, ...],
    attempt_manifest: AttemptManifest,
    attempt_readiness: AttemptReadiness,
    finalization_client_result: ClientResult,
    candidate_output_manifest: CandidateOutputManifest,
    shard_winner: ShardWinner,
) -> None:
    assert single_node_plan.run_id == "run-single"
    assert multi_node_plan.run_id == "run-001"
    assert client_result.run_id == "run-001"
    assert run_manifest.run_id == "run-single"
    assert len(shard_manifests) == 1
    assert attempt_manifest.attempt_id == "attempt-0001"
    assert attempt_readiness.revision == 1
    assert finalization_client_result.outcome.value == "complete"
    assert candidate_output_manifest.winner_eligible
    assert shard_winner.attempt_id == "attempt-0001"
