# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from pydantic import BaseModel, Field, JsonValue, ValidationError

from data_designer.config.column_configs import LLMStructuredColumnConfig, LLMTextColumnConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.storage_schema import JSON_STORAGE_KEY, JsonStoragePlan, compile_json_storage_schema


class ToolCall(BaseModel):
    name: str
    arguments: dict[str, JsonValue] = Field(json_schema_extra={JSON_STORAGE_KEY: "json"})


class Oracle(BaseModel):
    calls: list[ToolCall]


@pytest.mark.parametrize("model", [True, False])
def test_pydantic_and_json_schema_compile_identically(model: bool) -> None:
    schema = Oracle.model_json_schema()
    original = deepcopy(schema)
    column = LLMStructuredColumnConfig(
        name="oracle", prompt="Generate", model_alias="test", output_format=Oracle if model else schema
    )
    cleaned, paths = compile_json_storage_schema(column.output_format)
    assert paths == [["calls", None, "arguments"]]
    assert JSON_STORAGE_KEY not in cleaned["$defs"]["ToolCall"]["properties"]["arguments"]
    assert column.output_format == original
    assert schema == original


def test_deep_paths_and_escaped_reference_names() -> None:
    node: dict[str, Any] = {"$ref": "#/$defs/a~1b~0c"}
    for index in range(25):
        node = {"type": "object", "properties": {f"level{index}": {"type": "array", "items": node}}}
    node["$defs"] = {"a/b~c": {"type": "object", JSON_STORAGE_KEY: "json"}}
    _, paths = compile_json_storage_schema(node)
    assert paths == [[part for index in reversed(range(25)) for part in (f"level{index}", None)]]


def test_reference_site_annotation_and_alias() -> None:
    class Aliased(BaseModel):
        internal: ToolCall = Field(alias="external", json_schema_extra={JSON_STORAGE_KEY: "json"})

    _, paths = compile_json_storage_schema(Aliased.model_json_schema())
    assert paths == [["external"]]


def test_nullable_annotation_applies_to_entire_value() -> None:
    schema = {"anyOf": [{"type": "object"}, {"type": "null"}], JSON_STORAGE_KEY: "json"}
    cleaned, paths = compile_json_storage_schema(schema)
    assert paths == [[]]
    assert cleaned == {"anyOf": [{"type": "object"}, {"type": "null"}]}


@pytest.mark.parametrize("keyword", ["const", "enum", "default", "examples"])
def test_literals_are_not_treated_as_schemas(keyword: str) -> None:
    value = {JSON_STORAGE_KEY: "not a storage declaration"}
    schema = {keyword: value if keyword in ("const", "default") else [value]}
    assert compile_json_storage_schema(schema) == (schema, [])


@pytest.mark.parametrize("keyword", ["anyOf", "oneOf", "allOf", "prefixItems"])
def test_reject_branch_dependent_encoding(keyword: str) -> None:
    with pytest.raises(InvalidConfigError, match="enclosing"):
        compile_json_storage_schema({keyword: [{JSON_STORAGE_KEY: "json"}]})


@pytest.mark.parametrize("keyword", ["additionalProperties", "if", "then", "else", "not", "contains"])
def test_reject_unsupported_encoding_locations(keyword: str) -> None:
    with pytest.raises(InvalidConfigError, match="enclosing"):
        compile_json_storage_schema({keyword: {JSON_STORAGE_KEY: "json"}})


@pytest.mark.parametrize("value", [None, False, 1, "jsno", {"encoding": "json"}])
def test_invalid_annotation_value(value: Any) -> None:
    with pytest.raises(InvalidConfigError, match="must be 'json'"):
        compile_json_storage_schema({JSON_STORAGE_KEY: value})


def test_overlapping_annotations_are_rejected() -> None:
    with pytest.raises(InvalidConfigError, match="Overlapping"):
        compile_json_storage_schema({JSON_STORAGE_KEY: "json", "properties": {"nested": {JSON_STORAGE_KEY: "json"}}})


@pytest.mark.parametrize("reference", ["https://example.com/schema", "#/$defs/missing"])
def test_reject_external_or_missing_references(reference: str) -> None:
    with pytest.raises(InvalidConfigError):
        compile_json_storage_schema({"properties": {"a": {"$ref": reference}, "b": {JSON_STORAGE_KEY: "json"}}})


def test_recursive_schema_requires_encoded_ancestor() -> None:
    schema = {
        "$ref": "#/$defs/Node",
        "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}, "value": {JSON_STORAGE_KEY: "json"}}}},
    }
    with pytest.raises(InvalidConfigError, match="Recursive"):
        compile_json_storage_schema(schema)
    plain_recursive = {"$ref": "#/$defs/Node", "$defs": {"Node": {"properties": {"child": {"$ref": "#/$defs/Node"}}}}}
    plain_recursive[JSON_STORAGE_KEY] = "json"
    assert compile_json_storage_schema(plain_recursive)[1] == [[]]


def test_column_encoding_is_explicit_and_schema_overlap_rejected() -> None:
    assert (
        LLMTextColumnConfig(
            name="text", prompt="Generate", model_alias="test", storage_encoding="json"
        ).storage_encoding
        == "json"
    )
    with pytest.raises(InvalidConfigError, match="overlaps"):
        LLMStructuredColumnConfig(
            name="oracle", prompt="Generate", model_alias="test", output_format=Oracle, storage_encoding="json"
        )
    with pytest.raises(ValidationError):
        LLMTextColumnConfig(name="text", prompt="Generate", model_alias="test", storage_encoding="invalid")


@pytest.mark.parametrize("paths", [[[], []], [["a"], ["a", "b"]], [[None], [None, "x"]]])
def test_metadata_cannot_contain_overlapping_paths(paths: list[list[str | None]]) -> None:
    with pytest.raises(ValidationError, match="Overlapping"):
        JsonStoragePlan(columns={"payload": paths})


def test_storage_annotation_rejects_unsupported_dialect() -> None:
    with pytest.raises(InvalidConfigError, match="2020-12"):
        compile_json_storage_schema({"$schema": "http://json-schema.org/draft-07/schema#", JSON_STORAGE_KEY: "json"})


def test_storage_plan_version_is_not_coerced_from_boolean() -> None:
    with pytest.raises(ValidationError):
        JsonStoragePlan.model_validate({"version": True, "columns": {}})
