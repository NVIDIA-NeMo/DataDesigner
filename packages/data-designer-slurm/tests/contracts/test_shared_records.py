# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from data_designer.slurm.benchmark import BenchmarkManifest, BenchmarkReport
from data_designer.slurm.client import ClientResult
from data_designer.slurm.contracts import (
    ArtifactReference as CommonArtifactReference,
)
from data_designer.slurm.contracts import (
    RecordRange as CommonRecordRange,
)
from data_designer.slurm.contracts import (
    ResumeWorkspace as CommonResumeWorkspace,
)
from data_designer.slurm.planning import (
    ArtifactReference as PlanningArtifactReference,
)
from data_designer.slurm.planning import (
    RecordRange as PlanningRecordRange,
)
from data_designer.slurm.planning import (
    ResumeWorkspace as PlanningResumeWorkspace,
)

GOLDEN_DIR = Path(__file__).parent / "golden"


def test_planning_reexports_common_contract_types() -> None:
    assert PlanningArtifactReference is CommonArtifactReference
    assert PlanningRecordRange is CommonRecordRange
    assert PlanningResumeWorkspace is CommonResumeWorkspace


@pytest.mark.parametrize("actual_records", [0, 25])
def test_client_result_allows_partial_and_failure_facts_to_differ(actual_records: int) -> None:
    partial = {
        "schema_version": 1,
        "run_id": "run-001",
        "shard_id": "shard-00000",
        "attempt_id": "attempt-0001",
        "completed_at": "2026-08-19T12:00:00Z",
        "requested_records": 50,
        "actual_records": actual_records,
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

    assert ClientResult.model_validate_json(json.dumps(partial)).actual_records == actual_records
    assert ClientResult.model_validate_json(json.dumps(failed)).dataset_path is None


@pytest.mark.parametrize(
    "mutation",
    [
        {"actual_records": 51},
        {"outcome": "complete", "actual_records": 49},
        {"outcome": "failed", "candidate_output_manifest": {"path": "/x", "sha256": "a" * 64}},
        {"outcome": "failed", "candidate_output_manifest": None, "error_code": None},
        {"effective_resume_mode": "always"},
        {"early_shutdown": None},
        {"early_shutdown": True},
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


def test_benchmark_report_rejects_unknown_or_incomplete_recommendations() -> None:
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

    payload["recommendations"] = [{"kind": "pareto", "case_id": "case"}]
    with pytest.raises(ValidationError, match="successful feasible"):
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


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actual_records", 999),
        ("generation_seconds", 0),
        ("wall_seconds", 0),
        ("rows_per_second", 0),
    ],
)
def test_successful_benchmark_case_requires_complete_positive_output(field: str, value: int) -> None:
    payload = json.loads((GOLDEN_DIR / "benchmark_report.json").read_text())
    payload["cases"][0][field] = value

    with pytest.raises(ValidationError):
        BenchmarkReport.model_validate_json(json.dumps(payload))


def test_benchmark_report_allows_pareto_frontier_but_singleton_minima() -> None:
    payload = json.loads((GOLDEN_DIR / "benchmark_report.json").read_text())
    second_case = deepcopy(payload["cases"][0])
    second_case.update(case_id="second-case", child_run_id="second-run", topology_digest="f" * 64)
    payload["cases"].append(second_case)
    payload["recommendations"] = [
        {"kind": "pareto", "case_id": payload["cases"][0]["case_id"]},
        {"kind": "pareto", "case_id": second_case["case_id"]},
    ]

    assert len(BenchmarkReport.model_validate_json(json.dumps(payload)).recommendations) == 2

    payload["recommendations"] = [
        {"kind": "minimum_jobs", "case_id": payload["cases"][0]["case_id"]},
        {"kind": "minimum_jobs", "case_id": second_case["case_id"]},
    ]
    with pytest.raises(ValidationError, match="minimum"):
        BenchmarkReport.model_validate_json(json.dumps(payload))


def test_benchmark_report_rejects_duplicate_child_runs() -> None:
    payload = json.loads((GOLDEN_DIR / "benchmark_report.json").read_text())
    payload["cases"][1]["child_run_id"] = payload["cases"][0]["child_run_id"]

    with pytest.raises(ValidationError, match="child run IDs"):
        BenchmarkReport.model_validate_json(json.dumps(payload))
