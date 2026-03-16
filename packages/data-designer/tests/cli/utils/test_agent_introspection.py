# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.utils.agent_introspection import (
    AgentContextError,
    discover_family_types,
    get_builder_api,
    get_family_catalog,
    get_family_schema,
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


def test_get_family_schema_returns_json_schema_payload() -> None:
    schema_payload = get_family_schema("validator", "code")

    assert schema_payload["family"] == "validators"
    assert schema_payload["type_name"] == "code"
    assert schema_payload["class_name"] == "CodeValidatorParams"
    assert schema_payload["import_path"] == "data_designer.config.CodeValidatorParams"
    assert schema_payload["schema"]["title"] == "CodeValidatorParams"


def test_get_family_schema_includes_schema_view() -> None:
    schema_payload = get_family_schema("columns", "llm-text")

    view = schema_payload["schema_view"]
    assert view.class_name == "LLMTextColumnConfig"
    assert schema_payload["import_path"] == "data_designer.config.LLMTextColumnConfig"
    assert "column_type" not in [field.name for field in view.fields]
    assert "allow_resize" not in [field.name for field in view.fields]
    assert "name" in [field.name for field in view.fields]
    assert view.summary is not None


def test_get_family_schema_raises_for_unknown_type() -> None:
    with pytest.raises(AgentContextError) as exc_info:
        get_family_schema("validators", "does-not-exist")

    assert exc_info.value.code == "unknown_type"
    assert exc_info.value.details["family"] == "validators"
    assert "code" in exc_info.value.details["available_types"]


def test_discover_family_types_returns_pydantic_classes() -> None:
    types_map = discover_family_types("columns")

    assert types_map
    assert all(hasattr(cls, "model_fields") for cls in types_map.values())


def test_get_family_spec_returns_discriminator_field() -> None:
    spec = get_family_spec("columns")

    assert spec.name == "columns"
    assert spec.discriminator_field == "column_type"


def test_get_builder_api_includes_docstrings() -> None:
    builder_api = get_builder_api()

    assert builder_api["class_name"] == "DataDesignerConfigBuilder"
    assert builder_api["import_path"] == "data_designer.config.DataDesignerConfigBuilder"
    assert builder_api["methods"]
    assert all("docstring" in method for method in builder_api["methods"])


def test_get_types_returns_all_families_when_no_family_given() -> None:
    data = get_types(None)

    assert "families" in data
    assert "items" in data
    assert len(data["families"]) > 0
    assert all(family["family"] in data["items"] for family in data["families"])


def test_get_types_returns_single_family() -> None:
    data = get_types("columns")

    assert data["family"] == "columns"
    assert isinstance(data["items"], list)
    assert len(data["items"]) > 0


def test_get_operations_returns_all_commands() -> None:
    operations = get_operations()

    assert len(operations) == 6
    assert all(
        "name" in operation and "command_pattern" in operation and "description" in operation
        for operation in operations
    )
