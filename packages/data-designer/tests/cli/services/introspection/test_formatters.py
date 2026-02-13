# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.formatters import (
    format_method_info_json,
    format_method_info_text,
    format_model_schema_json,
    format_model_schema_text,
    format_overview_text,
    format_type_list_text,
)
from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo
from data_designer.cli.services.introspection.pydantic_inspector import FieldDetail, ModelSchema

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_field(name: str = "my_field", type_str: str = "str", description: str = "A field") -> FieldDetail:
    return FieldDetail(name=name, type_str=type_str, description=description)


def _make_schema(
    class_name: str = "TestModel",
    description: str = "A test model.",
    type_key: str | None = None,
    type_value: str | None = None,
    fields: list[FieldDetail] | None = None,
) -> ModelSchema:
    return ModelSchema(
        class_name=class_name,
        description=description,
        type_key=type_key,
        type_value=type_value,
        fields=fields or [_make_field()],
    )


def _make_method(
    name: str = "do_thing",
    signature: str = "do_thing(x: int) -> str",
    description: str = "Does a thing.",
    return_type: str = "str",
    parameters: list[ParamInfo] | None = None,
) -> MethodInfo:
    return MethodInfo(
        name=name,
        signature=signature,
        description=description,
        return_type=return_type,
        parameters=parameters or [ParamInfo(name="x", type_str="int", default=None, description="An integer")],
    )


# ---------------------------------------------------------------------------
# format_model_schema_text
# ---------------------------------------------------------------------------


def test_format_model_schema_text_basic() -> None:
    schema = _make_schema()
    text = format_model_schema_text(schema)
    assert "TestModel:" in text
    assert "description: A test model." in text
    assert "my_field:" in text
    assert "type: str" in text


def test_format_model_schema_text_with_type_key() -> None:
    schema = _make_schema(type_key="column_type", type_value="llm-text")
    text = format_model_schema_text(schema)
    assert "column_type: llm-text" in text


def test_format_model_schema_text_with_nested_schema() -> None:
    nested = _make_schema(class_name="NestedModel", description="Nested.")
    outer_field = FieldDetail(
        name="child",
        type_str="NestedModel",
        description="A nested model",
        nested_schema=nested,
    )
    schema = _make_schema(fields=[outer_field])
    text = format_model_schema_text(schema)
    assert "schema (NestedModel):" in text


def test_format_model_schema_text_with_enum_values() -> None:
    field = FieldDetail(
        name="color",
        type_str="ColorEnum",
        description="Pick a color",
        enum_values=["RED", "GREEN", "BLUE"],
    )
    schema = _make_schema(fields=[field])
    text = format_model_schema_text(schema)
    assert "values: [RED, GREEN, BLUE]" in text


# ---------------------------------------------------------------------------
# format_model_schema_json
# ---------------------------------------------------------------------------


def test_format_model_schema_json_basic() -> None:
    schema = _make_schema()
    result = format_model_schema_json(schema)
    assert isinstance(result, dict)
    assert result["class_name"] == "TestModel"
    assert result["description"] == "A test model."
    assert isinstance(result["fields"], list)
    assert len(result["fields"]) == 1
    assert result["fields"][0]["name"] == "my_field"


def test_format_model_schema_json_with_type_key() -> None:
    schema = _make_schema(type_key="column_type", type_value="sampler")
    result = format_model_schema_json(schema)
    assert result["column_type"] == "sampler"


def test_format_model_schema_json_with_nested() -> None:
    nested = _make_schema(class_name="Inner", description="Inner model.")
    outer_field = FieldDetail(
        name="inner",
        type_str="Inner",
        description="Nested",
        nested_schema=nested,
    )
    schema = _make_schema(fields=[outer_field])
    result = format_model_schema_json(schema)
    inner_field = result["fields"][0]
    assert "schema" in inner_field
    assert inner_field["schema"]["class_name"] == "Inner"


# ---------------------------------------------------------------------------
# format_method_info_text
# ---------------------------------------------------------------------------


def test_format_method_info_text_basic() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods)
    assert "do_thing(x: int) -> str" in text
    assert "Does a thing." in text
    assert "Parameters:" in text


def test_format_method_info_text_with_class_name() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods, class_name="MyBuilder")
    assert "MyBuilder Methods:" in text


def test_format_method_info_text_no_class_name() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods, class_name=None)
    assert "Methods:" not in text


# ---------------------------------------------------------------------------
# format_method_info_json
# ---------------------------------------------------------------------------


def test_format_method_info_json_basic() -> None:
    methods = [_make_method()]
    result = format_method_info_json(methods)
    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "do_thing"
    assert entry["signature"] == "do_thing(x: int) -> str"
    assert entry["return_type"] == "str"
    assert "description" in entry
    assert "parameters" in entry


def test_format_method_info_json_multiple_methods() -> None:
    methods = [
        _make_method(name="method_a", signature="method_a() -> None", return_type="None", parameters=[]),
        _make_method(name="method_b"),
    ]
    result = format_method_info_json(methods)
    assert len(result) == 2
    names = [e["name"] for e in result]
    assert "method_a" in names
    assert "method_b" in names


# ---------------------------------------------------------------------------
# format_type_list_text
# ---------------------------------------------------------------------------


def test_format_type_list_text_basic() -> None:
    class FakeA:
        pass

    class FakeB:
        pass

    items: dict[str, type] = {"alpha": FakeA, "beta": FakeB}
    text = format_type_list_text(items, "type_name", "class_name")
    assert "type_name" in text
    assert "class_name" in text
    assert "alpha" in text
    assert "FakeA" in text
    assert "beta" in text
    assert "FakeB" in text


def test_format_type_list_text_alignment() -> None:
    class C:
        pass

    items: dict[str, type] = {"short": C, "very_long_name": C}
    text = format_type_list_text(items, "Type", "Class")
    lines = text.strip().split("\n")
    # Header + separator + 2 data rows
    assert len(lines) == 4


def test_format_type_list_text_empty() -> None:
    text = format_type_list_text({}, "Type", "Class")
    assert "(no items)" in text


# ---------------------------------------------------------------------------
# format_overview_text
# ---------------------------------------------------------------------------


def test_format_overview_text_contains_header() -> None:
    type_counts = {"Column types": 5, "Sampler types": 3}
    methods = [_make_method()]
    text = format_overview_text(type_counts, methods)
    assert "Data Designer API Overview" in text


def test_format_overview_text_contains_type_counts() -> None:
    type_counts = {"Column types": 5, "Sampler types": 3}
    methods = [_make_method()]
    text = format_overview_text(type_counts, methods)
    assert "Type Counts:" in text
    assert "Column types:" in text
    assert "5" in text
    assert "Sampler types:" in text
    assert "3" in text


def test_format_overview_text_contains_builder_methods() -> None:
    type_counts = {"Column types": 5}
    methods = [_make_method(name="add_column")]
    text = format_overview_text(type_counts, methods)
    assert "Builder Methods" in text
    assert "add_column(...)" in text


def test_format_overview_text_contains_quick_start() -> None:
    type_counts = {"Column types": 1}
    text = format_overview_text(type_counts, [])
    assert "Quick Start Commands:" in text
    assert "agent-context columns --list" in text
