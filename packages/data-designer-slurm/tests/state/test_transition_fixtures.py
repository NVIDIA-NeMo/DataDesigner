# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from itertools import pairwise
from pathlib import Path

from data_designer.slurm.state import AttemptReadiness, validate_readiness_transition

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"
TRANSITION_FILES = (
    "single_node_pending_readiness.json",
    "single_node_starting_readiness.json",
    "single_node_readiness.json",
    "single_node_failed_readiness.json",
    "single_node_stopped_readiness.json",
)


def test_single_node_readiness_transition_fixture_is_canonical() -> None:
    records = tuple(
        AttemptReadiness.model_validate_json((GOLDEN_DIRECTORY / filename).read_text()) for filename in TRANSITION_FILES
    )

    assert tuple(record.revision for record in records) == (1, 2, 3, 4, 5)
    for previous, current in pairwise(records):
        assert validate_readiness_transition(previous, current) is current


def test_endpoint_publication_failure_transition_fixture_is_canonical() -> None:
    starting = AttemptReadiness.model_validate_json(
        (GOLDEN_DIRECTORY / "single_node_starting_readiness.json").read_text()
    )
    failed = AttemptReadiness.model_validate_json(
        (GOLDEN_DIRECTORY / "single_node_publication_failed_readiness.json").read_text()
    )

    assert validate_readiness_transition(starting, failed) is failed
