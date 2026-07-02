# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from datetime import datetime, timezone

from data_designer.engine.models.telemetry import (
    DeploymentTypeEnum,
    InferenceEvent,
    NemoSourceEnum,
    QueuedEvent,
    TaskStatusEnum,
    build_payload,
)


def test_nvidia_internal_deployment_type_uses_schema_version_1_9() -> None:
    event = InferenceEvent(
        nemo_source=NemoSourceEnum.DATADESIGNER,
        task="batch",
        task_status=TaskStatusEnum.SUCCESS,
        deployment_type=DeploymentTypeEnum("nvidia-internal"),
        model="test-model",
    )
    payload = build_payload(
        [QueuedEvent(event=event, timestamp=datetime(2026, 7, 2, tzinfo=timezone.utc))],
        source_client_version="test",
    )

    assert payload["eventSchemaVer"] == "1.9"
    assert payload["events"][0]["parameters"]["deploymentType"] == "nvidia-internal"
