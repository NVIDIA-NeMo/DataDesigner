# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from data_designer.engine.models.telemetry import (
    DeploymentTypeEnum,
    InferenceEvent,
    NemoSourceEnum,
    QueuedEvent,
    TaskStatusEnum,
    build_payload,
    get_nemo_deployment_type,
)


@pytest.mark.parametrize(
    "deployment_type",
    (DeploymentTypeEnum.NVIDIA_INTERNAL, DeploymentTypeEnum.SLURM, DeploymentTypeEnum.SLURM_NVIDIA_INTERNAL),
)
def test_supported_deployment_type_uses_schema_version_1_9(deployment_type: DeploymentTypeEnum) -> None:
    event = InferenceEvent(
        nemo_source=NemoSourceEnum.DATADESIGNER,
        task="batch",
        task_status=TaskStatusEnum.SUCCESS,
        deployment_type=deployment_type,
        model="test-model",
    )
    payload = build_payload(
        [QueuedEvent(event=event, timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc))],
        source_client_version="test",
    )

    assert payload["eventSchemaVer"] == "1.9"
    assert payload["events"][0]["parameters"]["deploymentType"] == deployment_type.value


@pytest.mark.parametrize(
    ("value", "expected"),
    (("slurm", DeploymentTypeEnum.SLURM), ("Slurm-nvidia-internal", DeploymentTypeEnum.SLURM_NVIDIA_INTERNAL)),
)
def test_slurm_environment_deployment_type_is_recognized(
    monkeypatch: pytest.MonkeyPatch, value: str, expected: DeploymentTypeEnum
) -> None:
    monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", value)

    assert get_nemo_deployment_type() is expected


def test_unrecognized_env_deployment_type_defaults_to_undefined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NEMO_DEPLOYMENT_TYPE", "unrecognized")

    assert get_nemo_deployment_type() == DeploymentTypeEnum.UNDEFINED
