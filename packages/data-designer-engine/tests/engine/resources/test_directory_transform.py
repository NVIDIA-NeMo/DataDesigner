# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_designer.config.seed_source import DirectoryListingTransform
from data_designer.engine.resources.directory_transform import (
    DirectoryTransformError,
    create_default_directory_transform_registry,
    create_directory_transform_context,
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
    normalized_records = transform.normalize(context=create_directory_transform_context(tmp_path))

    assert normalized_records == [
        {
            "source_kind": "stub_directory_transform",
            "root_path": str(tmp_path),
            "matched_file_count": 1,
            "matched_file_names": ["seed.jsonl"],
        }
    ]


def test_directory_transform_context_exposes_rooted_filesystem(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    context = create_directory_transform_context(tmp_path)

    assert sorted(context.fs.find("", withdirs=False)) == ["alpha.txt", "nested/beta.txt"]
    with context.fs.open("nested/beta.txt", "r", encoding="utf-8") as handle:
        assert handle.read() == "beta"


def test_directory_listing_transform_matches_file_names_recursively(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")

    transform = create_default_directory_transform_registry().create_transform(
        DirectoryListingTransform(file_pattern="*.txt", recursive=True)
    )
    normalized_records = transform.normalize(context=create_directory_transform_context(tmp_path))

    assert [record["relative_path"] for record in normalized_records] == ["alpha.txt", "nested/beta.txt"]


def test_directory_listing_transform_can_disable_recursive_walk(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")

    transform = create_default_directory_transform_registry().create_transform(
        DirectoryListingTransform(file_pattern="*.txt", recursive=False)
    )
    normalized_records = transform.normalize(context=create_directory_transform_context(tmp_path))

    assert [record["relative_path"] for record in normalized_records] == ["alpha.txt"]


def test_directory_listing_transform_matches_file_pattern_case_sensitively(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    transform = create_default_directory_transform_registry().create_transform(
        DirectoryListingTransform(file_pattern="*.TXT", recursive=True)
    )

    with pytest.raises(DirectoryTransformError, match="No files matched file_pattern '\\*\\.TXT'"):
        transform.normalize(context=create_directory_transform_context(tmp_path))


def test_directory_listing_transform_owns_directory_matching_options(tmp_path: Path) -> None:
    (tmp_path / "alpha.txt").write_text("alpha", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "beta.txt").write_text("beta", encoding="utf-8")
    (tmp_path / "nested" / "gamma.md").write_text("gamma", encoding="utf-8")

    transform = create_default_directory_transform_registry().create_transform(
        DirectoryListingTransform(file_pattern="*.txt", recursive=False)
    )

    normalized_records = transform.normalize(context=create_directory_transform_context(tmp_path))

    assert normalized_records == [
        {
            "source_kind": "directory_file",
            "source_path": str(tmp_path / "alpha.txt"),
            "relative_path": "alpha.txt",
            "file_name": "alpha.txt",
        }
    ]
