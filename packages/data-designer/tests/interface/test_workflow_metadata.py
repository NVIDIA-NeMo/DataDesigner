# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

import data_designer.interface as interface
from data_designer.interface import (
    CompletedWorkflowStageMetadata,
    FailedWorkflowStageMetadata,
    RunningWorkflowStageMetadata,
    SkippedWorkflowStageMetadata,
    WorkflowMetadata,
    WorkflowStageMetadata,
)


@pytest.fixture
def base_stage_metadata() -> dict[str, Any]:
    return {
        "index": 0,
        "name": "base",
        "stage_dir": "stage-0-base",
        "depends_on": [],
        "allow_empty": False,
        "on_success_version": None,
        "output_processors": [],
        "output": "final",
        "sampling_strategy": "ordered",
        "selection_strategy": None,
    }


@pytest.fixture
def started_stage_metadata() -> dict[str, Any]:
    return {
        "fingerprint": "abc123",
        "num_records_requested": 2,
        "seeded_from_stage": None,
        "seed_path": None,
        "config": {"columns": []},
    }


def test_workflow_metadata_models_are_public() -> None:
    assert interface.WorkflowMetadata is WorkflowMetadata
    assert interface.WorkflowStageMetadata is WorkflowStageMetadata
    assert interface.RunningWorkflowStageMetadata is RunningWorkflowStageMetadata
    assert interface.FailedWorkflowStageMetadata is FailedWorkflowStageMetadata
    assert interface.CompletedWorkflowStageMetadata is CompletedWorkflowStageMetadata
    assert interface.SkippedWorkflowStageMetadata is SkippedWorkflowStageMetadata
    assert issubclass(RunningWorkflowStageMetadata, WorkflowStageMetadata)


@pytest.mark.parametrize(
    ("status_fields", "expected_type"),
    [
        ({"status": "running"}, RunningWorkflowStageMetadata),
        ({"status": "failed", "duration_sec": 0.25}, FailedWorkflowStageMetadata),
        (
            {
                "status": "completed",
                "num_records_actual": 2,
                "output_records": 2,
                "output_seed_path": "stage-0-base/parquet-files",
                "callback_output_path": None,
                "stage_output_override_path": None,
                "output_processor_output_path": None,
                "duration_sec": 1.5,
            },
            CompletedWorkflowStageMetadata,
        ),
        (
            {
                "status": "completed_empty",
                "num_records_actual": 0,
                "output_records": 0,
                "output_seed_path": "stage-0-base/parquet-files",
                "callback_output_path": None,
                "stage_output_override_path": None,
                "output_processor_output_path": None,
                "duration_sec": 1.5,
            },
            CompletedWorkflowStageMetadata,
        ),
        (
            {"status": "skipped_empty_upstream", "upstream_stage": "base"},
            SkippedWorkflowStageMetadata,
        ),
    ],
)
def test_workflow_metadata_supports_stage_statuses(
    base_stage_metadata: dict[str, Any],
    started_stage_metadata: dict[str, Any],
    status_fields: dict[str, Any],
    expected_type: type,
) -> None:
    stage = base_stage_metadata | status_fields
    if status_fields["status"] != "skipped_empty_upstream":
        stage |= started_stage_metadata

    metadata = WorkflowMetadata.model_validate({"name": "example", "library_version": "0.9.2", "stages": [stage]})

    assert isinstance(metadata.stages[0], expected_type)
    restored = WorkflowMetadata.model_validate_json(metadata.model_dump_json())
    assert restored == metadata


def test_workflow_metadata_supports_legacy_completed_stage(
    base_stage_metadata: dict[str, Any],
    started_stage_metadata: dict[str, Any],
) -> None:
    stage = (
        base_stage_metadata
        | started_stage_metadata
        | {
            "status": "completed",
            "num_records_actual": 2,
            "output_records": 2,
            "output_seed_path": "/tmp/artifacts/example/stage-0-base/parquet-files",
            "callback_output_path": None,
            "output_processor_output_path": None,
            "duration_sec": 1.5,
        }
    )

    metadata = WorkflowMetadata.model_validate({"name": "example", "library_version": "0.9.2", "stages": [stage]})

    assert metadata.stages[0].stage_output_override_path is None
    assert "stage_output_override_path" not in metadata.model_dump(mode="json", exclude_unset=True)["stages"][0]


def test_workflow_metadata_preserves_extra_fields(
    base_stage_metadata: dict[str, Any],
    started_stage_metadata: dict[str, Any],
) -> None:
    stage = base_stage_metadata | started_stage_metadata | {"status": "running", "stage_extension": {"value": 1}}
    metadata = WorkflowMetadata.model_validate(
        {
            "name": "example",
            "library_version": "0.9.2",
            "stages": [stage],
            "workflow_extension": True,
        }
    )

    payload = metadata.model_dump(mode="json", exclude_unset=True)

    assert payload["workflow_extension"] is True
    assert payload["stages"][0]["stage_extension"] == {"value": 1}


@pytest.mark.parametrize(
    "stage_fields",
    [
        {"status": "unknown"},
        {"status": "running"},
        {"status": "failed", "duration_sec": "invalid"},
        {"status": "completed"},
        {"status": "skipped_empty_upstream"},
    ],
)
def test_workflow_metadata_rejects_invalid_stage_metadata(
    base_stage_metadata: dict[str, Any],
    stage_fields: dict[str, Any],
) -> None:
    with pytest.raises(ValidationError):
        WorkflowMetadata.model_validate(
            {
                "name": "example",
                "library_version": "0.9.2",
                "stages": [base_stage_metadata | stage_fields],
            }
        )
