# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from data_designer.engine.resources.directory_transform import create_default_directory_transform_registry
from data_designer.engine.testing.stubs import StubDirectoryTransformConfig, plugin_directory_transform
from data_designer.plugins.plugin import PluginType


def test_create_default_directory_transform_registry_loads_plugins(tmp_path: Path) -> None:
    matched_file = tmp_path / "seed.jsonl"
    matched_file.write_text("{}\n", encoding="utf-8")

    with patch(
        "data_designer.engine.resources.directory_transform.PluginRegistry.get_plugins",
        side_effect=lambda plugin_type: [plugin_directory_transform]
        if plugin_type == PluginType.DIRECTORY_TRANSFORM
        else [],
    ):
        registry = create_default_directory_transform_registry()

    transform = registry.create_transform(StubDirectoryTransformConfig())
    normalized_records = transform.normalize(root_path=tmp_path, matched_files=[matched_file])

    assert normalized_records == [
        {
            "source_kind": "stub_directory_transform",
            "root_path": str(tmp_path),
            "matched_file_count": 1,
            "matched_file_names": ["seed.jsonl"],
        }
    ]
