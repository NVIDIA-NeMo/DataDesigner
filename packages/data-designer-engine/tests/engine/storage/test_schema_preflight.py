# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, JsonValue

from data_designer.config.column_configs import CustomColumnConfig, LLMStructuredColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.storage_encoding import encode_storage_table
from data_designer.config.utils.storage_schema import JSON_STORAGE_KEY, JsonStoragePlan, compile_json_storage_schema
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.storage.schema_preflight import find_storage_schema_risks
from data_designer.engine.validation import (
    ViolationLevel,
    ViolationType,
    validate_data_designer_config,
    validate_parquet_storage,
)


def _column(schema: Any, **kwargs: Any) -> LLMStructuredColumnConfig:
    return LLMStructuredColumnConfig(
        name="oracle", prompt="Generate", model_alias="test_model", output_format=schema, **kwargs
    )


def _record(child: dict[str, Any], *, required: bool = True) -> dict[str, Any]:
    return {"type": "object", "properties": {"arguments": child}, "required": ["arguments"] if required else []}


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "object", "additionalProperties": False},
        {"type": "object", "maxProperties": 0},
        {"const": {}},
        {"const": [1, "two", True]},
        _record({"type": "object", "additionalProperties": False}),
        {"type": "array", "minItems": 1, "items": {"type": "object", "additionalProperties": False}},
    ],
)
def test_proven_native_failures_are_errors(schema: dict[str, Any]) -> None:
    violations = validate_parquet_storage([_column(schema)])
    assert any(v.level == ViolationLevel.ERROR for v in violations)
    assert all(v.column == "oracle" and v.type == ViolationType.PARQUET_STORAGE for v in violations)
    assert all("x-data-designer-storage" in v.message and "storage_encoding='json'" in v.message for v in violations)


@pytest.mark.parametrize(
    "schema",
    [
        {},
        {"type": "object"},
        {"type": "object", "additionalProperties": {"type": "integer"}},
        {"type": "object", "patternProperties": {".*": {"type": "integer"}}},
        {"type": "array"},
        {"type": "array", "items": {}},
        {"type": ["integer", "string"]},
        {"anyOf": [{"type": "integer"}, {"type": "string"}]},
        {"oneOf": [{"const": {}}, {"type": "string"}]},
        {"allOf": [{"type": "object", "additionalProperties": False}]},
        {"type": ["object", "null"], "additionalProperties": False},
        _record({"type": "object", "additionalProperties": False}, required=False),
        {"type": "array", "items": {"type": "object", "additionalProperties": False}},
        {"enum": [{}, {"present": 1}]},
        {"enum": [{}, {"nested": {}}]},
        {"enum": [1, "two"]},
        {"enum": [True, 1]},
        {"enum": [{}, *[{"present": 1}] * 70]},
        {
            "$ref": "#/$defs/Node",
            "$defs": {"Node": {"type": "object", "properties": {"next": {"$ref": "#/$defs/Node"}}}},
        },
        {"type": "array", "prefixItems": [{"type": "integer"}, {"type": "string"}]},
        {"if": {"type": "object"}, "then": {"maxProperties": 0}},
    ],
)
def test_value_dependent_risks_warn_without_blocking(schema: dict[str, Any]) -> None:
    violations = validate_parquet_storage([_column(schema)])
    assert violations
    assert all(v.level == ViolationLevel.WARNING for v in violations)


@pytest.mark.parametrize(
    "schema",
    [
        {"type": "string"},
        {"type": ["number", "integer", "null"]},
        {"type": ["string", "null"]},
        {"anyOf": [{"type": "integer"}, {"type": "number"}, {"type": "null"}]},
        {"oneOf": [{"type": "string"}, {"type": "null"}]},
        _record({"type": "integer"}),
        {"type": "array", "items": {"type": "string"}},
        {"const": {"value": [1, 2]}},
        {"enum": [1, 2, 3]},
        {"const": None},
        {"const": []},
        {"type": "string", "examples": [{"empty": {}}], "$defs": {"Unused": {"const": {}}}},
    ],
)
def test_supported_shapes_do_not_warn(schema: dict[str, Any]) -> None:
    assert validate_parquet_storage([_column(schema)]) == []


@pytest.mark.parametrize("root_encoding", [False, True])
def test_explicit_json_storage_suppresses_findings(root_encoding: bool) -> None:
    schema = _record({"type": "object", "additionalProperties": False})
    if root_encoding:
        column = _column(schema, storage_encoding="json")
    else:
        schema["properties"]["arguments"][JSON_STORAGE_KEY] = "json"
        column = _column(schema)
    assert validate_parquet_storage([column]) == []


def test_annotated_child_does_not_hide_unrelated_risk() -> None:
    schema = {
        "type": "object",
        "properties": {
            "encoded": {"type": "object", JSON_STORAGE_KEY: "json"},
            "unencoded": {"type": "object"},
        },
    }
    risks = find_storage_schema_risks(schema)
    assert [risk.path for risk in risks] == [("unencoded",)]


def test_local_refs_aliases_and_nested_array_paths() -> None:
    class Call(BaseModel):
        args: dict[str, JsonValue] = Field(alias="arguments")

    class Oracle(BaseModel):
        calls: list[Call]

    violations = validate_parquet_storage([_column(Oracle)])
    assert any("$['calls'][]['arguments']" in v.message for v in violations)
    assert all(v.level == ViolationLevel.WARNING for v in violations)


def test_refs_resolve_escaped_json_pointer_names() -> None:
    schema = {"$ref": "#/$defs/a~1b~0c", "$defs": {"a/b~c": {"const": {}}}}
    assert find_storage_schema_risks(schema)[0].definite


def test_large_enum_is_never_rejected_from_partial_probe() -> None:
    schema = {"enum": [{}] * 64 + [{"present": 1}]}
    risks = find_storage_schema_risks(schema)
    assert risks and not any(risk.definite for risk in risks)


def test_individually_failing_enum_values_can_complement_one_another() -> None:
    values = [{"a": {}, "b": {"x": 1}}, {"a": {"x": 1}, "b": {}}]
    # Each value alone contains an unwritable empty struct; together Arrow can
    # infer both nested field types. Preflight must therefore only warn.
    with pa.BufferOutputStream() as sink:
        pq.write_table(pa.Table.from_pydict({"value": values}), sink)
    risks = find_storage_schema_risks({"enum": values})
    assert risks and not any(risk.definite for risk in risks)


def test_scalar_enums_do_not_load_arrow() -> None:
    with patch("data_designer.engine.storage.schema_preflight.lazy") as imports:
        assert find_storage_schema_risks({"enum": list(range(1000))}) == []
        assert imports.mock_calls == []


def test_custom_columns_are_not_speculatively_rejected() -> None:
    @custom_column_generator()
    def generate(row: dict[str, Any]) -> dict[str, Any]:
        return row

    assert validate_parquet_storage([CustomColumnConfig(name="custom", generator_function=generate)]) == []


def test_validation_pipeline_includes_storage_diagnostics() -> None:
    violations = validate_data_designer_config(
        columns=[_column(_record({"const": {}}))], processor_configs=[], allowed_references=[]
    )
    assert any(v.type == ViolationType.PARQUET_STORAGE and v.level == ViolationLevel.ERROR for v in violations)


def test_proven_failure_stops_before_model_calls(stub_resource_provider: Mock, stub_model_configs: list) -> None:
    config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config.add_column(_column(_record({"const": {}})))
    with patch("data_designer.engine.dataset_builders.dataset_builder.run_readiness_check") as readiness:
        with pytest.raises(InvalidConfigError, match="validation errors"):
            DatasetBuilder(config.build(), stub_resource_provider)
    readiness.assert_not_called()
    stub_resource_provider.model_registry.arun_health_check.assert_not_called()


def test_warning_allows_builder_and_shows_actionable_diagnostic(
    stub_resource_provider: Mock, stub_model_configs: list, capsys: pytest.CaptureFixture[str]
) -> None:
    config = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config.add_column(_column(_record({"type": "object"})))
    DatasetBuilder(config.build(), stub_resource_provider)
    output = capsys.readouterr().out
    assert "PARQUET_STORAGE" in output and "x-data-designer-storage" in output
    assert "oracle" in output and "arguments" in output


@pytest.mark.parametrize("min_items", [0, 1, 2])
def test_positional_array_item_risks_are_warning_only(min_items: int) -> None:
    schema = {
        "type": "array",
        "minItems": min_items,
        "prefixItems": [{"type": "integer"}],
        "items": {"type": "object", "additionalProperties": False},
    }
    risks = find_storage_schema_risks(schema)
    assert risks and all(not risk.definite for risk in risks)
    if min_items < 2:
        with pa.BufferOutputStream() as sink:
            pq.write_table(pa.Table.from_pydict({"value": [[1]]}), sink)


@pytest.mark.parametrize("dialect", ["http://json-schema.org/draft-07/schema#", "https://example.com/future-draft"])
def test_unverified_dialects_warn_without_interpreting_ref_siblings(dialect: str) -> None:
    schema = {
        "$schema": dialect,
        "$ref": "#/definitions/Text",
        "type": "object",
        "additionalProperties": False,
        "definitions": {"Text": {"type": "string"}},
    }
    risks = find_storage_schema_risks(schema)
    assert risks and all(not risk.definite and "dialect" in risk.reason for risk in risks)


def test_nested_resource_references_cannot_produce_a_false_fatal_proof() -> None:
    schema = {
        "type": "object",
        "$defs": {"Value": {"type": "object", "additionalProperties": False}},
        "properties": {
            "wrapper": {
                "$id": "https://example.com/child",
                "$defs": {"Value": {"type": "integer"}},
                "type": "object",
                "properties": {"value": {"$ref": "#/$defs/Value"}},
                "required": ["value"],
            }
        },
        "required": ["wrapper"],
    }
    risks = find_storage_schema_risks(schema)
    assert risks and all(not risk.definite for risk in risks)
    assert any(risk.path == ("wrapper",) and "$id" in risk.reason for risk in risks)


def test_reference_crossing_a_nested_resource_warns_instead_of_guessing_scope() -> None:
    schema = {
        "$ref": "#/$defs/Resource/$defs/Value",
        "$defs": {
            "Resource": {
                "$id": "https://example.com/child",
                "$defs": {"Value": {"type": "object", "additionalProperties": False}},
            }
        },
    }
    risks = find_storage_schema_risks(schema)
    assert risks and all(not risk.definite for risk in risks)


@pytest.mark.parametrize("reference", ["#/$defs/a%20b/anyOf/0", "#/$defs/a~1b~0c"])
def test_preflight_uses_the_shared_local_pointer_semantics(reference: str) -> None:
    schema = {
        "$ref": reference,
        "$defs": {
            "a b": {"anyOf": [{"type": "object", "additionalProperties": False}]},
            "a/b~c": {"type": "object", "additionalProperties": False},
        },
    }
    assert any(risk.definite for risk in find_storage_schema_risks(schema))


def test_root_recursive_reference_warns_without_recursing_forever() -> None:
    risks = find_storage_schema_risks({"type": "object", "properties": {"child": {"$ref": "#"}}})
    assert risks and all(not risk.definite for risk in risks)


@pytest.mark.parametrize("value", [-(2**63), 2**63 - 1, 2**63, 2**64 - 1])
def test_native_scalar_integer_boundaries_match_the_pandas_writer(value: int) -> None:
    with pa.BufferOutputStream() as sink:
        pd.DataFrame({"value": [value]}).to_parquet(sink, index=False)
    assert find_storage_schema_risks({"const": value}) == []
    assert find_storage_schema_risks({"enum": [value]}) == []


@pytest.mark.parametrize("value", [-(2**63) - 1, 2**64])
def test_constants_beyond_native_integer_ranges_remain_fatal(value: int) -> None:
    with pa.BufferOutputStream() as sink:
        with pytest.raises((OverflowError, pa.ArrowException)):
            pd.DataFrame({"value": [value]}).to_parquet(sink, index=False)
    risks = find_storage_schema_risks({"const": value})
    assert risks and any(risk.definite for risk in risks)


@pytest.mark.parametrize("kind", ["string", "integer", "number", "boolean", "null"])
@pytest.mark.parametrize(
    "keywords",
    [
        {"additionalProperties": False},
        {"maxProperties": 0},
        {"properties": {"bad": {"const": {}}}, "required": ["bad"]},
        {"items": {"const": {}}, "minItems": 1},
        {"prefixItems": [{"const": {}}], "minItems": 1},
    ],
)
def test_inactive_container_keywords_do_not_warn_for_scalar_or_null_types(kind: str, keywords: dict[str, Any]) -> None:
    assert find_storage_schema_risks({"type": kind, **keywords}) == []


@pytest.mark.parametrize(
    "types",
    [None, ["object", "null"], ["object", "integer"], ["array", "null"], ["array", "string"]],
)
def test_unknown_nullable_and_heterogeneous_container_types_cannot_prove_failure(types: list[str] | None) -> None:
    schema = {
        "properties": {"bad": {"const": {}}},
        "required": ["bad"],
        "items": {"const": {}},
        "minItems": 1,
    }
    if types is not None:
        schema["type"] = types
    risks = find_storage_schema_risks(schema)
    assert risks and all(not risk.definite for risk in risks)


@pytest.mark.parametrize("location", ["root", "nested", "reference"])
def test_encoded_descendants_make_empty_object_schemas_writable(location: str) -> None:
    empty = {
        "type": "object",
        "maxProperties": 0,
        "properties": {"args": {JSON_STORAGE_KEY: "json"}},
    }
    value: dict[str, Any] = {}
    if location == "nested":
        schema = {"type": "object", "properties": {"result": empty}, "required": ["result"]}
        value = {"result": {}}
    elif location == "reference":
        schema = {"$ref": "#/$defs/Empty", "$defs": {"Empty": empty}}
    else:
        schema = empty
    Draft202012Validator(schema).validate(value)
    _, paths = compile_json_storage_schema(schema)
    plan = JsonStoragePlan(columns={"oracle": paths})
    table = encode_storage_table(pd.DataFrame({"oracle": [value]}), plan)
    with pa.BufferOutputStream() as sink:
        pq.write_table(table, sink)
    assert validate_parquet_storage([_column(schema)]) == []


@pytest.mark.parametrize("min_items", [2, 3])
def test_positional_array_elements_can_supply_fields_for_empty_trailing_items(min_items: int) -> None:
    schema = {
        "type": "array",
        "prefixItems": [{"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}],
        "items": {"const": {}},
        "minItems": min_items,
    }
    value = [{"x": 1}, *[{} for _ in range(min_items - 1)]]
    Draft202012Validator(schema).validate(value)
    with pa.BufferOutputStream() as sink:
        pd.DataFrame({"oracle": [value]}).to_parquet(sink, index=False)
    violations = validate_parquet_storage([_column(schema)])
    assert violations
    assert all(violation.level == ViolationLevel.WARNING for violation in violations)
