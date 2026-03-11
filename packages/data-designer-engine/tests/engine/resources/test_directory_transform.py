# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_designer.engine.resources.directory_transform import (
    DirectoryTransformError,
    create_default_directory_transform_registry,
    discover_directory_files,
)
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


def test_discover_directory_files_matches_file_names_recursively(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")

    matched_files = discover_directory_files(root_path=tmp_path, file_pattern="*.txt", recursive=True)

    assert [str(path.relative_to(tmp_path)) for path in matched_files] == ["alpha.txt", "nested/beta.txt"]


def test_discover_directory_files_can_disable_recursive_walk(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    matched_files = discover_directory_files(root_path=tmp_path, file_pattern="*.txt", recursive=False)

    assert [str(path.relative_to(tmp_path)) for path in matched_files] == ["alpha.txt"]


def test_discover_directory_files_rejects_matches_that_resolve_outside_root(tmp_path: Path) -> None:
    outside_file = tmp_path.parent / "outside-seed.txt"
    outside_file.write_text("outside", encoding="utf-8")
    try:
        (tmp_path / "outside-seed.txt").symlink_to(outside_file)
    except OSError:
        pytest.skip("Symlink creation is not supported in this environment")

    with pytest.raises(DirectoryTransformError, match="resolves outside the directory seed root"):
        discover_directory_files(root_path=tmp_path, file_pattern="*.txt", recursive=True)
