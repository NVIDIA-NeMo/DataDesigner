# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

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

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
GOLDEN_MODELS: tuple[tuple[str, type[StateRecord]], ...] = (
    ("run_manifest.json", RunManifest),
    ("shard_manifest.json", ShardManifest),
    ("successful_attempt.json", AttemptManifest),
    ("single_node_readiness.json", AttemptReadiness),
    ("multi_node_readiness.json", AttemptReadiness),
    ("failed_attempt.json", AttemptManifest),
    ("stale_readiness.json", AttemptReadiness),
    ("accounting_lag.json", SchedulerObservation),
    ("candidate_output.json", CandidateOutputManifest),
    ("shard_winner.json", ShardWinner),
    ("collection_plan.json", CollectionPlan),
)


@pytest.mark.parametrize(("filename", "model"), GOLDEN_MODELS)
def test_golden_record_round_trip_is_deterministic(filename: str, model: type[StateRecord]) -> None:
    serialized = (GOLDEN_DIRECTORY / filename).read_text()

    record = model.model_validate_json(serialized)
    direct_record = model(**record.model_dump(mode="python"))

    assert direct_record == record
    assert record.serialize_json() == serialized
    assert model.model_validate_json(record.serialize_json()) == record
    assert len(record.compute_sha256()) == 64
