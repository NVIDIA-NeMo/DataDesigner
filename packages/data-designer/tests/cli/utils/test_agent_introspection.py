# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.utils.agent_introspection import (
    discover_family_types,
    get_config_package_path,
    get_family_catalog,
    get_family_source_file,
    get_family_spec,
    get_operations,
    get_types,
)


def test_get_family_catalog_accepts_singular_family_names() -> None:
    assert get_family_catalog("validator") == get_family_catalog("validators")


def test_get_family_catalog_returns_sorted_type_names() -> None:
    catalog = get_family_catalog("columns")
    assert catalog
    assert [item["type_name"] for item in catalog] == sorted(item["type_name"] for item in catalog)


def test_get_family_catalog_includes_description() -> None:
    catalog = get_family_catalog("columns")
    item = next(i for i in catalog if i["type_name"] == "llm-text")

    assert "Configuration for text generation" in item["description"]


def test_get_family_source_file_returns_relative_path() -> None:
    path = get_family_source_file("columns")

    assert path == "data_designer/config/column_configs.py"


def test_discover_family_types_returns_pydantic_classes() -> None:
    types_map = discover_family_types("columns")

    assert types_map
    assert all(hasattr(cls, "model_fields") for cls in types_map.values())


def test_get_family_spec_returns_discriminator_field() -> None:
    spec = get_family_spec("columns")

    assert spec.name == "columns"
    assert spec.discriminator_field == "column_type"


def test_get_types_returns_all_families_when_no_family_given() -> None:
    data = get_types(None)

    assert "families" in data
    assert "items" in data
    assert len(data["families"]) > 0
    assert all(f["family"] in data["items"] for f in data["families"])
    assert all("file" in f for f in data["families"])


def test_get_types_returns_single_family() -> None:
    data = get_types("columns")

    assert data["family"] == "columns"
    assert data["file"].endswith(".py")
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0


def test_get_operations_returns_all_commands() -> None:
    ops = get_operations()

    assert len(ops) == 4
    assert all("name" in op and "command_pattern" in op and "description" in op for op in ops)


def test_get_config_package_path_returns_config_dir() -> None:
    path = get_config_package_path()

    assert "data_designer" in path
    assert "config" in path
