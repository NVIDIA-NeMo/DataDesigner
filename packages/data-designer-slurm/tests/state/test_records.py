# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from data_designer.slurm.state import (
    ArtifactReference,
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputManifest,
    RecordRange,
    RunManifest,
)

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"


def _golden_payload(filename: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIRECTORY / filename).read_text())


def test_direct_construction_and_json_loading_produce_identical_record() -> None:
    record = RunManifest(
        schema_version=1,
        run_id="run-direct",
        created_at=datetime(2026, 8, 18, 12, tzinfo=timezone.utc),
        authored_config=ArtifactReference(path="/workspace/run/authored.json", sha256="a" * 64),
        resolved_plan=ArtifactReference(path="/workspace/run/plan.json", sha256="b" * 64),
        shard_count=2,
    )

    assert RunManifest.model_validate_json(record.serialize_json()) == record
    assert record.serialize_json() == RunManifest.model_validate_json(record.serialize_json()).serialize_json()


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    (
        ("schema_version", 2),
        ("run_id", "contains/slash"),
        ("run_id", " leading-space"),
    ),
)
def test_run_manifest_rejects_invalid_version_and_identity(
    field_name: str,
    invalid_value: object,
) -> None:
    payload = _golden_payload("run_manifest.json")
    payload[field_name] = invalid_value

    with pytest.raises(ValidationError):
        RunManifest.model_validate_json(json.dumps(payload))


def test_records_reject_unknown_fields() -> None:
    payload = _golden_payload("run_manifest.json")
    payload["unreviewed"] = True

    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        RunManifest.model_validate_json(json.dumps(payload))


def test_records_require_explicit_schema_version() -> None:
    payload = _golden_payload("run_manifest.json")
    payload.pop("schema_version")

    with pytest.raises(ValidationError, match="Field required"):
        RunManifest.model_validate_json(json.dumps(payload))


def test_artifact_reference_rejects_invalid_digest_and_path() -> None:
    with pytest.raises(ValidationError):
        ArtifactReference(path="/workspace/plan.json", sha256="A" * 64)
    with pytest.raises(ValidationError, match="parent-directory"):
        ArtifactReference(path="/workspace/../private/plan.json", sha256="a" * 64)
    with pytest.raises(ValidationError, match="absolute"):
        ArtifactReference(path="relative/plan.json", sha256="a" * 64)


def test_records_are_frozen() -> None:
    record = RunManifest.model_validate_json((GOLDEN_DIRECTORY / "run_manifest.json").read_text())

    with pytest.raises(ValidationError, match="Instance is frozen"):
        record.run_id = "another-run"


def test_record_range_is_half_open_and_non_empty() -> None:
    record_range = RecordRange(start_index=10, end_index_exclusive=15)

    assert record_range.record_count == 5
    with pytest.raises(ValidationError, match="greater than start_index"):
        RecordRange(start_index=10, end_index_exclusive=10)


@pytest.mark.parametrize(
    ("filename", "mutation"),
    (
        (
            "successful_attempt.json",
            {"candidate_output": None},
        ),
        (
            "successful_attempt.json",
            {"state": "running", "terminal_classification": "succeeded"},
        ),
        (
            "successful_attempt.json",
            {"state": "running", "terminal_classification": None, "scheduler": None},
        ),
    ),
)
def test_attempt_lifecycle_invariants(filename: str, mutation: dict[str, object]) -> None:
    payload = _golden_payload(filename)
    payload.update(mutation)

    with pytest.raises(ValidationError):
        AttemptManifest.model_validate_json(json.dumps(payload))


def test_ready_deployment_requires_all_backends_and_published_endpoint() -> None:
    payload = _golden_payload("single_node_readiness.json")
    deployment = payload["deployments"][0]
    deployment["ready_backends"] = 0

    with pytest.raises(ValidationError, match="every expected backend"):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_readiness_rejects_duplicate_deployment_identity() -> None:
    payload = _golden_payload("multi_node_readiness.json")
    payload["deployments"][1]["model_alias"] = payload["deployments"][0]["model_alias"]

    with pytest.raises(ValidationError, match="model aliases must be unique"):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_probe_evidence_is_bounded_and_single_line() -> None:
    payload = _golden_payload("single_node_readiness.json")
    payload["deployments"][0]["last_probe"]["redacted_message"] = "line one\nline two"

    with pytest.raises(ValidationError, match="control characters"):
        AttemptReadiness.model_validate_json(json.dumps(payload))

    payload["deployments"][0]["last_probe"]["redacted_message"] = "x" * 513
    with pytest.raises(ValidationError):
        AttemptReadiness.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("attempt_state", "deployment_state"),
    (
        ("stopped", "ready"),
        ("pending", "ready"),
        ("starting", "stopped"),
    ),
)
def test_attempt_readiness_rejects_contradictory_deployment_states(
    attempt_state: str,
    deployment_state: str,
) -> None:
    payload = _golden_payload("single_node_readiness.json")
    payload["state"] = attempt_state
    payload["deployments"][0]["state"] = deployment_state

    with pytest.raises(ValidationError):
        AttemptReadiness.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("state", "endpoint_publication"),
    (("pending", "pending"), ("stopped", "published")),
)
def test_pending_and_stopped_deployments_reject_ready_backends(
    state: str,
    endpoint_publication: str,
) -> None:
    payload = _golden_payload("single_node_readiness.json")
    payload["state"] = state
    payload["deployments"][0].update(
        {
            "state": state,
            "endpoint_publication": endpoint_publication,
            "ready_backends": 1,
        }
    )

    with pytest.raises(ValidationError, match="cannot have ready backends"):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_partial_candidate_is_never_winner_eligible() -> None:
    payload = _golden_payload("candidate_output.json")
    payload["actual_records"] = 99
    payload["outcome"] = "partial"
    payload["files"][0]["record_count"] = 99

    partial_candidate = CandidateOutputManifest.model_validate_json(json.dumps(payload))
    assert not partial_candidate.winner_eligible


def test_candidate_output_rejects_count_and_file_mismatches() -> None:
    payload = _golden_payload("candidate_output.json")
    payload["files"][0]["record_count"] = 99

    with pytest.raises(ValidationError, match="record counts"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))

    payload = _golden_payload("candidate_output.json")
    payload["files"][0]["relative_path"] = "../escaped.parquet"
    with pytest.raises(ValidationError, match="parent-directory"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))
