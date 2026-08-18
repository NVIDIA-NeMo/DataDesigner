# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from data_designer.slurm.state import (
    ArtifactReference,
    AttemptManifest,
    AttemptReadiness,
    CandidateOutputFile,
    CandidateOutputManifest,
    CollectionPlan,
    RunManifest,
    SchedulerObservation,
)

GOLDEN_DIRECTORY = Path(__file__).parent / "golden"


def _golden_payload(filename: str) -> dict[str, Any]:
    return json.loads((GOLDEN_DIRECTORY / filename).read_text())


@pytest.mark.parametrize(
    "path",
    (
        "/",
        "/workspace/control\ncharacter",
        "/workspace//unnormalized",
    ),
)
def test_absolute_artifact_paths_reject_unsafe_forms(path: str) -> None:
    with pytest.raises(ValidationError):
        ArtifactReference(path=path, sha256="a" * 64)


@pytest.mark.parametrize(
    "path",
    (
        "",
        "/absolute",
        "control\ncharacter",
        ".",
        "directory//unnormalized",
    ),
)
def test_relative_output_paths_reject_unsafe_forms(path: str) -> None:
    with pytest.raises(ValidationError):
        CandidateOutputFile(
            relative_path=path,
            sha256="a" * 64,
            byte_size=0,
            record_count=0,
        )


@pytest.mark.parametrize(
    "created_at",
    (
        "2026-08-18T12:00:00",
        "2026-08-18T13:00:00+01:00",
    ),
)
def test_run_manifest_requires_utc_timestamp(created_at: str) -> None:
    payload = _golden_payload("run_manifest.json")
    payload["created_at"] = created_at

    with pytest.raises(ValidationError):
        RunManifest.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ({"updated_at": "2026-08-18T12:00:01Z"}, "must not precede"),
        ({"terminal_classification": "failed"}, "succeeded terminal classification"),
        ({"state": "failed", "terminal_classification": "succeeded"}, "failed attempts"),
    ),
)
def test_attempt_manifest_rejects_invalid_terminal_details(
    mutation: dict[str, object],
    message: str,
) -> None:
    payload = _golden_payload("successful_attempt.json")
    payload.update(mutation)

    with pytest.raises(ValidationError, match=message):
        AttemptManifest.model_validate_json(json.dumps(payload))


def test_candidate_output_rejects_remaining_invalid_inventory_forms() -> None:
    payload = _golden_payload("candidate_output.json")
    payload["actual_records"] = 101
    with pytest.raises(ValidationError, match="must not exceed"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))

    payload = _golden_payload("candidate_output.json")
    payload["outcome"] = "partial"
    with pytest.raises(ValidationError, match="outcome must be"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))

    payload = _golden_payload("candidate_output.json")
    duplicate_file = dict(payload["files"][0])
    duplicate_file["record_count"] = 0
    payload["files"].append(duplicate_file)
    with pytest.raises(ValidationError, match="paths must be unique"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))

    payload = _golden_payload("candidate_output.json")
    payload["files"] = []
    with pytest.raises(ValidationError, match="require at least one file"):
        CandidateOutputManifest.model_validate_json(json.dumps(payload))


def test_collection_plan_rejects_duplicate_shards_and_winner_paths() -> None:
    payload = _golden_payload("collection_plan.json")
    duplicate_shard = json.loads(json.dumps(payload["planned_shards"][0]))
    duplicate_shard["winner_manifest"]["path"] = "/workspace/runs/run-0001/shards/shard-0001/winner.json"
    payload["planned_shards"].append(duplicate_shard)
    with pytest.raises(ValidationError, match="shard IDs must be unique"):
        CollectionPlan.model_validate_json(json.dumps(payload))

    payload = _golden_payload("collection_plan.json")
    duplicate_winner = json.loads(json.dumps(payload["planned_shards"][0]))
    duplicate_winner["shard_id"] = "shard-0001"
    payload["planned_shards"].append(duplicate_winner)
    with pytest.raises(ValidationError, match="winner manifest paths must be unique"):
        CollectionPlan.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize(
    ("attempt_state", "deployment_updates", "message"),
    (
        ("ready", {"ready_backends": 2}, "must not exceed"),
        ("ready", {"endpoint_publication": "failed"}, "failed endpoint publication"),
        (
            "pending",
            {"state": "pending", "ready_backends": 0, "endpoint_publication": "published"},
            "pending endpoint publication",
        ),
        ("starting", {"state": "starting"}, "must use the ready state"),
        ("ready", {"endpoint_publication": "pending"}, "published endpoint"),
        (
            "starting",
            {"state": "stopped", "ready_backends": 0},
            "cannot contain failed or stopped",
        ),
        ("starting", {}, "every deployment ready"),
        (
            "ready",
            {"state": "starting", "ready_backends": 0, "endpoint_publication": "pending"},
            "every deployment to be ready",
        ),
        ("failed", {}, "at least one failed deployment"),
    ),
)
def test_readiness_rejects_remaining_contradictory_states(
    attempt_state: str,
    deployment_updates: dict[str, object],
    message: str,
) -> None:
    payload = _golden_payload("single_node_readiness.json")
    payload["state"] = attempt_state
    payload["deployments"][0].update(deployment_updates)

    with pytest.raises(ValidationError, match=message):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_readiness_rejects_duplicate_deployment_names() -> None:
    payload = _golden_payload("multi_node_readiness.json")
    payload["deployments"][1]["deployment_name"] = payload["deployments"][0]["deployment_name"]

    with pytest.raises(ValidationError, match="deployment names must be unique"):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_readiness_rejects_probe_observation_after_snapshot() -> None:
    payload = _golden_payload("single_node_readiness.json")
    payload["deployments"][0]["last_probe"]["observed_at"] = "2026-08-18T12:03:00Z"

    with pytest.raises(ValidationError, match="later than the readiness snapshot"):
        AttemptReadiness.model_validate_json(json.dumps(payload))


def test_scheduler_observation_requires_consistent_deadline() -> None:
    payload = _golden_payload("accounting_lag.json")
    payload["reconciliation_deadline"] = None
    with pytest.raises(ValidationError, match="requires a reconciliation deadline"):
        SchedulerObservation.model_validate_json(json.dumps(payload))

    payload = _golden_payload("accounting_lag.json")
    payload["reconciliation_deadline"] = "2026-08-18T12:05:01Z"
    with pytest.raises(ValidationError, match="must not precede"):
        SchedulerObservation.model_validate_json(json.dumps(payload))

    payload = _golden_payload("accounting_lag.json")
    payload["state"] = "running"
    with pytest.raises(ValidationError, match="only accounting lag"):
        SchedulerObservation.model_validate_json(json.dumps(payload))

    payload = _golden_payload("accounting_lag.json")
    payload["reconciliation_deadline"] = "2026-08-18T13:10:02+01:00"
    with pytest.raises(ValidationError, match="must be in UTC"):
        SchedulerObservation.model_validate_json(json.dumps(payload))
