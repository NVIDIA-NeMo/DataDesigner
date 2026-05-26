# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.config.scheduling import SchedulingMetadata, SchedulingMetadataError


@pytest.mark.parametrize(
    "metadata",
    [
        SchedulingMetadata.local(),
        SchedulingMetadata.model("nvidia", "nemotron", "chat", weight=2),
        SchedulingMetadata.custom_model("plugin", "resource", "v1"),
    ],
)
def test_scheduling_metadata_accepts_normative_shapes(metadata: SchedulingMetadata) -> None:
    assert metadata.weight >= 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"identity": ["local", "default"]},
        {"weight": True},
        {"kind": "model", "identity": ("local", "default")},
        {"kind": "local", "identity": ("local", "default", "extra")},
        {"kind": "custom_model", "identity": ("custom_model", "plugin")},
    ],
)
def test_scheduling_metadata_rejects_non_normative_direct_construction(kwargs: dict[str, object]) -> None:
    with pytest.raises(SchedulingMetadataError):
        SchedulingMetadata(**kwargs)  # type: ignore[arg-type]
