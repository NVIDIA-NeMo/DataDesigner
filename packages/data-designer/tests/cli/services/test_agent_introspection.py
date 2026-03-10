# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.services.agent_introspection import (
    AgentIntrospectionError,
    get_builder_api,
    get_family_catalog,
    get_family_schema,
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


def test_get_family_schema_raises_for_unknown_type() -> None:
    with pytest.raises(AgentIntrospectionError) as exc_info:
        get_family_schema("validators", "does-not-exist")

    assert exc_info.value.code == "unknown_type"
    assert exc_info.value.details["family"] == "validators"
    assert "code" in exc_info.value.details["available_types"]


def test_get_builder_api_can_omit_docstrings() -> None:
    builder_api = get_builder_api(include_docstrings=False)

    assert builder_api["class_name"] == "DataDesignerConfigBuilder"
    assert builder_api["import_path"] == "data_designer.config.DataDesignerConfigBuilder"
    assert builder_api["methods"]
    assert all(method["docstring"] is None for method in builder_api["methods"])
