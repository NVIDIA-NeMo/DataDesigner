# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy

import pytest
from pydantic import ValidationError

from data_designer.slurm._contracts import ArtifactReference as CommonArtifactReference
from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.client import ClientResult
from data_designer.slurm.planning import ArtifactReference as PlanningArtifactReference


def test_planning_reexports_common_artifact_reference() -> None:
    assert PlanningArtifactReference is CommonArtifactReference


def test_client_result_allows_partial_and_failure_facts_to_differ() -> None:
    partial = {
        "schema_version": 1,
        "run_id": "run-001",
        "shard_id": "shard-00000",
        "attempt_id": "attempt-0001",
        "completed_at": "2026-08-19T12:00:00Z",
        "requested_records": 50,
        "actual_records": 25,
        "outcome": "partial",
        "dataset_path": "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0001/dataset",
        "early_shutdown": True,
        "requested_resume_mode": "if_possible",
        "effective_resume_mode": "never",
        "candidate_output_manifest": {
            "path": "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0001/output-manifest.json",
            "sha256": "a" * 64,
        },
    }
    failed = {
        "schema_version": 1,
        "run_id": "run-001",
        "shard_id": "shard-00000",
        "attempt_id": "attempt-0002",
        "completed_at": "2026-08-19T12:00:00Z",
        "requested_records": 50,
        "actual_records": None,
        "outcome": "failed",
        "early_shutdown": None,
        "requested_resume_mode": "if_possible",
        "effective_resume_mode": None,
        "error_code": "generation_error",
        "redacted_message": "generation failed",
    }

    assert ClientResult.model_validate_json(json.dumps(partial)).actual_records == 25
    assert ClientResult.model_validate_json(json.dumps(failed)).dataset_path is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"actual_records": 51},
        {"outcome": "complete", "actual_records": 49},
        {"outcome": "partial", "actual_records": 0},
        {"outcome": "failed", "candidate_output_manifest": {"path": "/x", "sha256": "a" * 64}},
        {"outcome": "failed", "candidate_output_manifest": None, "error_code": None},
        {"effective_resume_mode": "always"},
        {"early_shutdown": None},
        {"dataset_path": "/workspace/other/dataset"},
        {
            "candidate_output_manifest": {
                "path": "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0002/output-manifest.json",
                "sha256": "a" * 64,
            }
        },
        {"completed_at": "2026-08-19T12:00:00+01:00"},
        {"redacted_message": "bad\nmessage"},
    ],
)
def test_client_result_rejects_inconsistent_semantics(
    mutation: dict[str, object],
    client_result_payload: dict[str, object],
) -> None:
    payload = deepcopy(client_result_payload)
    payload.update(mutation)

    with pytest.raises(ValidationError):
        ClientResult.model_validate_json(json.dumps(payload))


@pytest.fixture
def client_result_payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "run_id": "run-001",
        "shard_id": "shard-00000",
        "attempt_id": "attempt-0001",
        "completed_at": "2026-08-19T12:00:00Z",
        "requested_records": 50,
        "actual_records": 50,
        "outcome": "complete",
        "dataset_path": "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0001/dataset",
        "early_shutdown": False,
        "requested_resume_mode": "never",
        "effective_resume_mode": "never",
        "candidate_output_manifest": {
            "path": "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0001/output-manifest.json",
            "sha256": "a" * 64,
        },
    }


@pytest.mark.parametrize(
    ("effective_resume_mode", "dataset_path"),
    [
        ("never", "/workspace/runs/run-001/shards/shard-00000/attempts/attempt-0001/dataset"),
        ("always", "/workspace/runs/run-001/shards/shard-00000/dataset"),
    ],
)
def test_if_possible_uses_effective_resume_dataset_location(
    effective_resume_mode: str,
    dataset_path: str,
    client_result_payload: dict[str, object],
) -> None:
    payload = deepcopy(client_result_payload)
    payload.update(
        requested_resume_mode="if_possible",
        effective_resume_mode=effective_resume_mode,
        dataset_path=dataset_path,
    )

    assert ClientResult.model_validate_json(json.dumps(payload)).dataset_path == dataset_path


def test_benchmark_manifest_rejects_duplicate_child_identity() -> None:
    payload = {
        "schema_version": 1,
        "benchmark_id": "bench",
        "benchmark_config": {"path": "/workspace/config.json", "sha256": "a" * 64},
        "children": [
            {
                "case_id": "case",
                "child_run_id": "run",
                "child_authored_config": {
                    "path": "/workspace/runs/run/authored-config.json",
                    "sha256": "b" * 64,
                },
            },
            {
                "case_id": "case",
                "child_run_id": "run-2",
                "child_authored_config": {
                    "path": "/workspace/runs/run-2/authored-config.json",
                    "sha256": "c" * 64,
                },
            },
        ],
    }

    with pytest.raises(ValidationError, match="case IDs"):
        BenchmarkManifest.model_validate_json(json.dumps(payload))


def test_benchmark_report_rejects_unknown_and_duplicate_recommendations() -> None:
    payload = {
        "schema_version": 1,
        "benchmark_id": "bench",
        "analysis_id": "analysis",
        "benchmark_manifest": {"path": "/workspace/benchmark.json", "sha256": "a" * 64},
        "created_at": "2026-08-19T12:00:00Z",
        "cases": [
            {
                "case_id": "case",
                "child_run_id": "run",
                "outcome": "pending",
                "topology_digest": "b" * 64,
                "requested_records": 100,
                "gpus_per_job": 8,
                "nodes_per_job": 1,
            }
        ],
        "recommendations": [{"kind": "pareto", "case_id": "missing"}],
    }

    with pytest.raises(ValidationError, match="unknown cases"):
        BenchmarkReport.model_validate_json(json.dumps(payload))

    payload["recommendations"] = [
        {"kind": "pareto", "case_id": "case"},
        {"kind": "pareto", "case_id": "case"},
    ]
    with pytest.raises(ValidationError, match="kinds"):
        BenchmarkReport.model_validate_json(json.dumps(payload))


def test_successful_benchmark_case_requires_metrics() -> None:
    payload = {
        "schema_version": 1,
        "benchmark_id": "bench",
        "analysis_id": "analysis",
        "benchmark_manifest": {"path": "/workspace/benchmark.json", "sha256": "a" * 64},
        "created_at": "2026-08-19T12:00:00Z",
        "cases": [
            {
                "case_id": "case",
                "child_run_id": "run",
                "outcome": "succeeded",
                "topology_digest": "b" * 64,
                "requested_records": 100,
                "gpus_per_job": 8,
                "nodes_per_job": 1,
            }
        ],
    }

    with pytest.raises(ValidationError, match="complete"):
        BenchmarkReport.model_validate_json(json.dumps(payload))
