# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from data_designer.slurm.state import (
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    CollectionPlan,
    RunManifest,
    SchedulerObservation,
    ShardManifest,
    ShardWinner,
    StateRecord,
)
from data_designer.slurm.state.storage import StateStorage

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
CONTRACT_GOLDEN_DIRECTORY = Path(__file__).parents[1] / "contracts" / "golden"
GOLDEN_MODELS: tuple[tuple[str, type[StateRecord]], ...] = (
    ("run_manifest.json", RunManifest),
    ("shard_manifest.json", ShardManifest),
    ("successful_attempt.json", AttemptManifest),
    ("single_node_pending_readiness.json", AttemptReadiness),
    ("single_node_starting_readiness.json", AttemptReadiness),
    ("single_node_readiness.json", AttemptReadiness),
    ("single_node_publication_failed_readiness.json", AttemptReadiness),
    ("single_node_failed_readiness.json", AttemptReadiness),
    ("single_node_stopped_readiness.json", AttemptReadiness),
    ("multi_node_readiness.json", AttemptReadiness),
    ("failed_attempt.json", AttemptManifest),
    ("stale_readiness.json", AttemptReadiness),
    ("accounting_lag.json", SchedulerObservation),
    ("candidate_output.json", CandidateOutputManifest),
    ("shard_winner.json", ShardWinner),
    ("collection_plan.json", CollectionPlan),
    ("collection_plan_v1.json", CollectionPlan),
)
PERSISTED_INTENT_FIXTURES = (
    CONTRACT_GOLDEN_DIRECTORY / "authored_run_single.json",
    CONTRACT_GOLDEN_DIRECTORY / "single_node_plan.json",
)
_COMPATIBILITY_FIXTURE_DIGEST = "222f3f38e5a4c33508fa51469780faef8f0dc817024bad676dc7866f67bbd284"


@pytest.mark.parametrize(("filename", "model"), GOLDEN_MODELS)
def test_state_golden_record_round_trip_is_deterministic(filename: str, model: type[StateRecord]) -> None:
    serialized = (GOLDEN_DIRECTORY / filename).read_text()

    record = model.model_validate_json(serialized)
    direct_record = model(**record.model_dump(mode="python"))

    assert direct_record == record
    assert record.serialize_json() == serialized
    assert model.model_validate_json(record.serialize_canonical_json()) == record
    assert model.model_validate_json(record.serialize_json()) == record
    assert record.compute_sha256() == hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def test_golden_artifact_references_hash_exact_persisted_bytes() -> None:
    candidate_bytes = (GOLDEN_DIRECTORY / "candidate_output.json").read_bytes()
    winner_bytes = (GOLDEN_DIRECTORY / "shard_winner.json").read_bytes()
    attempt = AttemptManifest.model_validate_json((GOLDEN_DIRECTORY / "successful_attempt.json").read_text())
    winner = ShardWinner.model_validate_json(winner_bytes)
    collection = CollectionPlan.model_validate_json((GOLDEN_DIRECTORY / "collection_plan.json").read_text())

    assert attempt.candidate_output is not None
    assert attempt.candidate_output.sha256 == hashlib.sha256(candidate_bytes).hexdigest()
    assert winner.candidate_manifest.sha256 == hashlib.sha256(candidate_bytes).hexdigest()
    assert collection.planned_shards[0].winner_manifest.sha256 == hashlib.sha256(winner_bytes).hexdigest()


@pytest.mark.parametrize("filename", ("collection_plan_v1.json", "collection_plan.json"))
def test_collection_storage_reads_both_plan_versions(tmp_path: Path, filename: str) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_bytes((GOLDEN_DIRECTORY / filename).read_bytes())
    plan_path.chmod(0o600)
    descriptor = os.open(tmp_path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        plan = StateStorage(tmp_path, "run-0001").read_record(descriptor, plan_path.name, plan_path, CollectionPlan)
    finally:
        os.close(descriptor)
    assert plan.compute_sha256() == hashlib.sha256(plan_path.read_bytes()).hexdigest()


def test_persisted_state_compatibility_fixtures_are_frozen() -> None:
    """Make persisted record byte changes an explicit compatibility decision."""
    fixture_bytes = b"".join(
        path.name.encode("utf-8") + b"\0" + path.read_bytes() for path in PERSISTED_INTENT_FIXTURES
    )
    fixture_bytes += b"".join(
        filename.encode("utf-8") + b"\0" + (GOLDEN_DIRECTORY / filename).read_bytes() for filename, _ in GOLDEN_MODELS
    )

    assert hashlib.sha256(fixture_bytes).hexdigest() == _COMPATIBILITY_FIXTURE_DIGEST
