# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field

from data_designer.cli.services.introspection.pydantic_inspector import (
    FieldDetail,
    ModelSchema,
    _extract_enum_class,
    _is_basemodel_subclass,
    _is_enum_subclass,
    build_model_schema,
    extract_nested_basemodel,
    format_type,
    get_brief_description,
    get_field_info,
)

# ---------------------------------------------------------------------------
# Test models / enums
# ---------------------------------------------------------------------------


class ColorEnum(str, Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class SizeEnum(str, Enum):
    SMALL = "small"
    LARGE = "large"


class InnerModel(BaseModel):
    x: int = 0
    y: str = "hello"


class OuterModel(BaseModel):
    """Outer model for testing."""

    plain: str = "foo"
    nested: InnerModel = Field(default_factory=InnerModel)
    my_enum: ColorEnum = ColorEnum.RED


class SelfRefModel(BaseModel):
    """A model that references itself (for cycle testing)."""

    name: str = ""
    child: SelfRefModel | None = None


class DeepA(BaseModel):
    val: int = 0
    b: DeepB | None = None


class DeepB(BaseModel):
    val: int = 0


# ---------------------------------------------------------------------------
# _is_basemodel_subclass
# ---------------------------------------------------------------------------


def test_is_basemodel_subclass_with_subclass() -> None:
    assert _is_basemodel_subclass(InnerModel) is True


def test_is_basemodel_subclass_with_basemodel_itself() -> None:
    assert _is_basemodel_subclass(BaseModel) is False


def test_is_basemodel_subclass_with_str() -> None:
    assert _is_basemodel_subclass(str) is False


def test_is_basemodel_subclass_with_enum() -> None:
    assert _is_basemodel_subclass(ColorEnum) is False


def test_is_basemodel_subclass_with_non_type() -> None:
    assert _is_basemodel_subclass("not a type") is False


# ---------------------------------------------------------------------------
# _is_enum_subclass
# ---------------------------------------------------------------------------


def test_is_enum_subclass_with_enum_subclass() -> None:
    assert _is_enum_subclass(ColorEnum) is True


def test_is_enum_subclass_with_enum_itself() -> None:
    assert _is_enum_subclass(Enum) is False


def test_is_enum_subclass_with_str() -> None:
    assert _is_enum_subclass(str) is False


def test_is_enum_subclass_with_non_type() -> None:
    assert _is_enum_subclass(42) is False


# ---------------------------------------------------------------------------
# _extract_enum_class
# ---------------------------------------------------------------------------


def test_extract_enum_class_direct_enum() -> None:
    assert _extract_enum_class(ColorEnum) is ColorEnum


def test_extract_enum_class_optional_enum() -> None:
    assert _extract_enum_class(ColorEnum | None) is ColorEnum


def test_extract_enum_class_annotated_enum() -> None:
    assert _extract_enum_class(Annotated[ColorEnum, "metadata"]) is ColorEnum


def test_extract_enum_class_non_enum() -> None:
    assert _extract_enum_class(str) is None


def test_extract_enum_class_none() -> None:
    assert _extract_enum_class(None) is None


# ---------------------------------------------------------------------------
# extract_nested_basemodel
# ---------------------------------------------------------------------------


def test_extract_nested_basemodel_direct() -> None:
    assert extract_nested_basemodel(InnerModel) is InnerModel


def test_extract_nested_basemodel_list() -> None:
    assert extract_nested_basemodel(list[InnerModel]) is InnerModel


def test_extract_nested_basemodel_optional() -> None:
    assert extract_nested_basemodel(InnerModel | None) is InnerModel


def test_extract_nested_basemodel_optional_list() -> None:
    assert extract_nested_basemodel(list[InnerModel] | None) is InnerModel


def test_extract_nested_basemodel_dict() -> None:
    assert extract_nested_basemodel(dict[str, InnerModel]) is InnerModel


def test_extract_nested_basemodel_annotated() -> None:
    assert extract_nested_basemodel(Annotated[InnerModel, "info"]) is InnerModel


def test_extract_nested_basemodel_discriminated_union_returns_none() -> None:
    """Unions of 2+ BaseModel subclasses should return None."""
    assert extract_nested_basemodel(InnerModel | OuterModel) is None


def test_extract_nested_basemodel_primitive_returns_none() -> None:
    assert extract_nested_basemodel(str) is None
    assert extract_nested_basemodel(int) is None


def test_extract_nested_basemodel_none_returns_none() -> None:
    assert extract_nested_basemodel(None) is None


def test_extract_nested_basemodel_basemodel_itself_returns_none() -> None:
    assert extract_nested_basemodel(BaseModel) is None


# ---------------------------------------------------------------------------
# format_type
# ---------------------------------------------------------------------------


def test_format_type_str() -> None:
    result = format_type(str)
    assert "str" in result


def test_format_type_int() -> None:
    result = format_type(int)
    assert "int" in result


def test_format_type_optional() -> None:
    result = format_type(str | None)
    assert "str" in result
    assert "None" in result


# ---------------------------------------------------------------------------
# get_brief_description
# ---------------------------------------------------------------------------


def test_get_brief_description_with_docstring() -> None:
    result = get_brief_description(OuterModel)
    assert result == "Outer model for testing."


def test_get_brief_description_without_docstring() -> None:
    result = get_brief_description(InnerModel)
    assert result == "No description available."


# ---------------------------------------------------------------------------
# get_field_info
# ---------------------------------------------------------------------------


def test_get_field_info_returns_field_details() -> None:
    fields = get_field_info(OuterModel)
    assert isinstance(fields, list)
    assert all(isinstance(f, FieldDetail) for f in fields)
    names = [f.name for f in fields]
    assert "plain" in names
    assert "nested" in names
    assert "my_enum" in names


def test_get_field_info_enum_values() -> None:
    fields = get_field_info(OuterModel)
    enum_field = next(f for f in fields if f.name == "my_enum")
    assert enum_field.enum_values is not None
    assert set(enum_field.enum_values) == {"RED", "GREEN", "BLUE"}


def test_get_field_info_non_enum_has_no_enum_values() -> None:
    fields = get_field_info(OuterModel)
    plain_field = next(f for f in fields if f.name == "plain")
    assert plain_field.enum_values is None


# ---------------------------------------------------------------------------
# build_model_schema
# ---------------------------------------------------------------------------


def test_build_model_schema_basic_structure() -> None:
    schema = build_model_schema(OuterModel)
    assert isinstance(schema, ModelSchema)
    assert schema.class_name == "OuterModel"
    assert schema.description == "Outer model for testing."
    assert len(schema.fields) == 3


def test_build_model_schema_with_type_key_and_value() -> None:
    schema = build_model_schema(OuterModel, type_key="column_type", type_value="test")
    assert schema.type_key == "column_type"
    assert schema.type_value == "test"


def test_build_model_schema_nested_expansion() -> None:
    schema = build_model_schema(OuterModel)
    nested_field = next(f for f in schema.fields if f.name == "nested")
    assert nested_field.nested_schema is not None
    assert nested_field.nested_schema.class_name == "InnerModel"
    nested_names = [f.name for f in nested_field.nested_schema.fields]
    assert "x" in nested_names
    assert "y" in nested_names


def test_build_model_schema_cycle_protection() -> None:
    schema = build_model_schema(SelfRefModel)
    child_field = next(f for f in schema.fields if f.name == "child")
    # SelfRefModel references itself: the first expansion should happen,
    # but the recursive child should NOT be expanded again (cycle detected).
    if child_field.nested_schema is not None:
        inner_child = next(
            (f for f in child_field.nested_schema.fields if f.name == "child"),
            None,
        )
        if inner_child is not None:
            assert inner_child.nested_schema is None


def test_build_model_schema_depth_limiting() -> None:
    schema = build_model_schema(DeepA, max_depth=1)
    b_field = next(f for f in schema.fields if f.name == "b")
    if b_field.nested_schema is not None:
        # At depth 1, nested model should not recurse further
        for f in b_field.nested_schema.fields:
            assert f.nested_schema is None
