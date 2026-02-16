# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.formatters import (
    format_imports_json,
    format_imports_text,
    format_interface_json,
    format_interface_text,
    format_method_info_json,
    format_method_info_text,
    format_model_schema_json,
    format_model_schema_text,
    format_namespace_json,
    format_namespace_text,
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
    assert "my_field: str" in text
    assert "[required]" in text


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
        enum_values=["red", "green", "blue"],
    )
    schema = _make_schema(fields=[field])
    text = format_model_schema_text(schema)
    assert "values: [red, green, blue]" in text


def test_format_model_schema_text_with_default() -> None:
    field = FieldDetail(
        name="count",
        type_str="int",
        description="A count",
        required=False,
        default="0",
    )
    schema = _make_schema(fields=[field])
    text = format_model_schema_text(schema)
    assert "count: int = 0" in text
    assert "[required]" not in text


def test_format_model_schema_text_with_constraints() -> None:
    field = FieldDetail(
        name="score",
        type_str="float",
        description="A score",
        required=False,
        default="0.5",
        constraints={"ge": 0.0, "le": 1.0},
    )
    schema = _make_schema(fields=[field])
    text = format_model_schema_text(schema)
    assert "constraints: ge=0.0, le=1.0" in text


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


def test_format_model_schema_json_includes_required_and_default() -> None:
    field = FieldDetail(name="val", type_str="int", description="Value", required=False, default="42")
    schema = _make_schema(fields=[field])
    result = format_model_schema_json(schema)
    f = result["fields"][0]
    assert f["required"] is False
    assert f["default"] == "42"


def test_format_model_schema_json_required_field_no_default() -> None:
    field = FieldDetail(name="val", type_str="str", description="Value", required=True)
    schema = _make_schema(fields=[field])
    result = format_model_schema_json(schema)
    f = result["fields"][0]
    assert f["required"] is True
    assert "default" not in f


def test_format_model_schema_json_includes_constraints() -> None:
    field = FieldDetail(
        name="score",
        type_str="float",
        description="Score",
        required=False,
        default="0.5",
        constraints={"ge": 0.0, "le": 1.0},
    )
    schema = _make_schema(fields=[field])
    result = format_model_schema_json(schema)
    f = result["fields"][0]
    assert f["constraints"] == {"ge": 0.0, "le": 1.0}


def test_format_model_schema_json_no_constraints_key_when_none() -> None:
    field = FieldDetail(name="val", type_str="str", description="Value")
    schema = _make_schema(fields=[field])
    result = format_model_schema_json(schema)
    f = result["fields"][0]
    assert "constraints" not in f


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
    assert "types columns" in text


# ---------------------------------------------------------------------------
# Namespace / code-structure formatters
# ---------------------------------------------------------------------------


def _make_namespace_data() -> dict:
    return {
        "paths": ["/fake/site-packages/data_designer"],
        "tree": {
            "name": "data_designer",
            "is_package": True,
            "children": [
                {
                    "name": "config",
                    "is_package": True,
                    "children": [
                        {"name": "column_configs", "is_package": False, "children": []},
                        {"name": "models", "is_package": False, "children": []},
                    ],
                },
                {
                    "name": "errors",
                    "is_package": False,
                    "children": [],
                },
            ],
        },
    }


def test_format_namespace_text_contains_tree_characters() -> None:
    text = format_namespace_text(_make_namespace_data())
    assert "├──" in text or "└──" in text
    assert "│" in text


def test_format_namespace_text_shows_install_path() -> None:
    text = format_namespace_text(_make_namespace_data())
    assert "Install path:" in text
    assert "/fake/site-packages/data_designer" in text


def test_format_namespace_text_packages_have_trailing_slash() -> None:
    text = format_namespace_text(_make_namespace_data())
    assert "config/" in text


def test_format_namespace_text_modules_have_py_extension() -> None:
    text = format_namespace_text(_make_namespace_data())
    assert "errors.py" in text
    assert "column_configs.py" in text


def test_format_namespace_text_contains_agent_guidance() -> None:
    text = format_namespace_text(_make_namespace_data())
    assert "Only read source files directly" in text


def test_format_namespace_json_returns_passthrough() -> None:
    data = _make_namespace_data()
    result = format_namespace_json(data)
    assert result is data


# ---------------------------------------------------------------------------
# Interface formatters
# ---------------------------------------------------------------------------


def _make_interface_data() -> tuple[list[tuple[str, list[MethodInfo]]], list[ModelSchema]]:
    methods_data = [
        ("DataDesigner", [_make_method(name="create", signature="create(...) -> DatasetCreationResults")]),
        ("DatasetCreationResults", [_make_method(name="load_dataset", signature="load_dataset() -> pd.DataFrame")]),
    ]
    schemas = [_make_schema(class_name="RunConfig", description="Runtime configuration.")]
    return methods_data, schemas


def test_format_interface_text_contains_class_names() -> None:
    methods_data, schemas = _make_interface_data()
    text = format_interface_text(methods_data, schemas)
    assert "DataDesigner" in text
    assert "DatasetCreationResults" in text
    assert "RunConfig" in text


def test_format_interface_text_contains_methods() -> None:
    methods_data, schemas = _make_interface_data()
    text = format_interface_text(methods_data, schemas)
    assert "create" in text
    assert "load_dataset" in text


def test_format_interface_text_contains_run_config_fields() -> None:
    methods_data, schemas = _make_interface_data()
    text = format_interface_text(methods_data, schemas)
    assert "my_field" in text


def test_format_interface_json_structure() -> None:
    methods_data, schemas = _make_interface_data()
    result = format_interface_json(methods_data, schemas)
    assert isinstance(result, dict)
    assert "methods" in result
    assert "schemas" in result
    assert "DataDesigner" in result["methods"]
    assert "DatasetCreationResults" in result["methods"]
    assert isinstance(result["schemas"], list)
    assert len(result["schemas"]) == 1


# ---------------------------------------------------------------------------
# Imports formatters
# ---------------------------------------------------------------------------


def _make_imports_data() -> dict[str, list[dict[str, str]]]:
    return {
        "Column Configs": [
            {"name": "LLMTextColumnConfig", "module": "data_designer.config"},
            {"name": "SamplerColumnConfig", "module": "data_designer.config"},
        ],
        "Interface": [
            {"name": "DataDesigner", "module": "data_designer.interface"},
        ],
    }


def test_format_imports_text_contains_recommended_imports() -> None:
    text = format_imports_text(_make_imports_data())
    assert "Recommended imports:" in text
    assert "import data_designer.config as dd" in text
    assert "from data_designer.interface import DataDesigner" in text


def test_format_imports_text_config_names_use_dd_prefix() -> None:
    text = format_imports_text(_make_imports_data())
    assert "dd.LLMTextColumnConfig" in text
    assert "dd.SamplerColumnConfig" in text
    assert "from data_designer.config import" not in text


def test_format_imports_text_interface_uses_from_import() -> None:
    text = format_imports_text(_make_imports_data())
    assert "from data_designer.interface import DataDesigner" in text


def test_format_imports_text_has_category_headers() -> None:
    text = format_imports_text(_make_imports_data())
    assert "Column Configs (2 names):" in text
    assert "Interface (1 name):" in text


def test_format_imports_json_structure() -> None:
    data = _make_imports_data()
    result = format_imports_json(data)
    assert isinstance(result, dict)
    assert "recommended_imports" in result
    assert "config_alias" in result
    assert result["config_alias"] == "dd"
    assert "categories" in result
    assert "Column Configs" in result["categories"]
    assert "Interface" in result["categories"]


def test_format_imports_json_category_structure() -> None:
    data = _make_imports_data()
    result = format_imports_json(data)
    config_cat = result["categories"]["Column Configs"]
    assert config_cat["module"] == "data_designer.config"
    assert config_cat["access_pattern"] == "dd.<name>"
    assert "LLMTextColumnConfig" in config_cat["names"]

    interface_cat = result["categories"]["Interface"]
    assert interface_cat["module"] == "data_designer.interface"
    assert "from data_designer.interface import <name>" in interface_cat["access_pattern"]
    assert "DataDesigner" in interface_cat["names"]


# ---------------------------------------------------------------------------
# Schema deduplication
# ---------------------------------------------------------------------------


def test_format_field_text_deduplicates_nested_schemas() -> None:
    """When seen_schemas is passed, second occurrence of a nested schema shows a back-reference."""
    nested = _make_schema(class_name="SharedNested", description="Shared nested model.")
    field1 = FieldDetail(name="field_a", type_str="SharedNested", description="First ref", nested_schema=nested)
    field2 = FieldDetail(name="field_b", type_str="SharedNested", description="Second ref", nested_schema=nested)

    schema1 = _make_schema(class_name="Model1", fields=[field1])
    schema2 = _make_schema(class_name="Model2", fields=[field2])

    seen: set[str] = set()
    text1 = format_model_schema_text(schema1, seen_schemas=seen)
    text2 = format_model_schema_text(schema2, seen_schemas=seen)

    assert "schema (SharedNested):" in text1
    assert "see SharedNested above" in text2
    assert "schema (SharedNested):" not in text2


def test_format_field_text_no_dedup_without_seen_set() -> None:
    """Without seen_schemas, nested schemas always expand fully."""
    nested = _make_schema(class_name="Inner", description="Inner model.")
    field1 = FieldDetail(name="x", type_str="Inner", description="Ref", nested_schema=nested)
    schema = _make_schema(fields=[field1])

    text = format_model_schema_text(schema)
    assert "schema (Inner):" in text
