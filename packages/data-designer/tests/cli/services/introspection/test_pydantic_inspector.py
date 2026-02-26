# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Annotated

import pytest
from pydantic import BaseModel, Field

from data_designer.cli.services.introspection.pydantic_inspector import (
    _default_to_json,
    _extract_constraints,
    _extract_enum_class,
    _extract_nested_basemodel,
    _is_basemodel_subclass,
    _is_enum_subclass,
    format_model_text,
    format_type,
    get_brief_description,
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


class DeepB(BaseModel):
    val: int = 0


class DeepA(BaseModel):
    val: int = 0
    b: DeepB | None = None


class SiblingNestedModel(BaseModel):
    first: InnerModel = Field(default_factory=InnerModel)
    second: InnerModel = Field(default_factory=InnerModel)


# Rebuild models that use forward references (required due to `from __future__ import annotations`)
SelfRefModel.model_rebuild()
DeepA.model_rebuild()


class RequiredFieldModel(BaseModel):
    """Model with required and optional fields for testing."""

    required_name: str
    optional_name: str = "default_val"


class ConstrainedModel(BaseModel):
    """Model with constrained fields for testing."""

    score: float = Field(default=0.5, ge=0.0, le=1.0)
    label: str = Field(default="", min_length=1, max_length=100)
    count: int = Field(default=0, gt=-1, lt=1000)


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
    assert _extract_nested_basemodel(InnerModel) is InnerModel


def test_extract_nested_basemodel_list() -> None:
    assert _extract_nested_basemodel(list[InnerModel]) is InnerModel


def test_extract_nested_basemodel_optional() -> None:
    assert _extract_nested_basemodel(InnerModel | None) is InnerModel


def test_extract_nested_basemodel_optional_list() -> None:
    assert _extract_nested_basemodel(list[InnerModel] | None) is InnerModel


def test_extract_nested_basemodel_dict() -> None:
    assert _extract_nested_basemodel(dict[str, InnerModel]) is InnerModel


def test_extract_nested_basemodel_annotated() -> None:
    assert _extract_nested_basemodel(Annotated[InnerModel, "info"]) is InnerModel


def test_extract_nested_basemodel_discriminated_union_returns_none() -> None:
    """Unions of 2+ BaseModel subclasses should return None."""
    assert _extract_nested_basemodel(InnerModel | OuterModel) is None


def test_extract_nested_basemodel_primitive_returns_none() -> None:
    assert _extract_nested_basemodel(str) is None
    assert _extract_nested_basemodel(int) is None


def test_extract_nested_basemodel_none_returns_none() -> None:
    assert _extract_nested_basemodel(None) is None


def test_extract_nested_basemodel_basemodel_itself_returns_none() -> None:
    assert _extract_nested_basemodel(BaseModel) is None


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
# _extract_constraints
# ---------------------------------------------------------------------------


def test_extract_constraints_from_constrained_model() -> None:
    score_info = ConstrainedModel.model_fields["score"]
    constraints = _extract_constraints(score_info)
    assert constraints is not None
    assert constraints["ge"] == 0.0
    assert constraints["le"] == 1.0


def test_extract_constraints_gt_lt() -> None:
    count_info = ConstrainedModel.model_fields["count"]
    constraints = _extract_constraints(count_info)
    assert constraints is not None
    assert constraints["gt"] == -1
    assert constraints["lt"] == 1000


def test_extract_constraints_string_lengths() -> None:
    label_info = ConstrainedModel.model_fields["label"]
    constraints = _extract_constraints(label_info)
    assert constraints is not None
    assert constraints["min_length"] == 1
    assert constraints["max_length"] == 100


def test_extract_constraints_none_for_unconstrained() -> None:
    x_info = InnerModel.model_fields["x"]
    assert _extract_constraints(x_info) is None


def test_extract_constraints_helper_with_no_metadata() -> None:
    """_extract_constraints returns None when field_info has no constraint metadata."""

    class FakeFieldInfo:
        metadata: list = []

    assert _extract_constraints(FakeFieldInfo()) is None


# ---------------------------------------------------------------------------
# format_model_text
# ---------------------------------------------------------------------------


def test_format_model_text_basic_structure() -> None:
    text = format_model_text(OuterModel)
    assert "OuterModel:" in text
    assert "description: Outer model for testing." in text
    assert "fields:" in text
    assert "plain:" in text
    assert "nested:" in text
    assert "my_enum:" in text


def test_format_model_text_with_type_key_and_value() -> None:
    text = format_model_text(OuterModel, type_key="column_type", type_value="test")
    assert "column_type: test" in text


def test_format_model_text_required_field() -> None:
    text = format_model_text(RequiredFieldModel)
    assert "required_name: str  [required]" in text


def test_format_model_text_optional_field_default() -> None:
    text = format_model_text(RequiredFieldModel)
    assert "optional_name: str = 'default_val'" in text
    assert "[required]" not in text.split("optional_name")[1].split("\n")[0]


def test_format_model_text_default_factory() -> None:
    text = format_model_text(OuterModel)
    assert "= InnerModel()" in text


def test_format_model_text_none_default() -> None:
    text = format_model_text(SelfRefModel)
    assert "child:" in text
    assert "= None" in text


def test_format_model_text_enum_default_uses_member_value() -> None:
    text = format_model_text(OuterModel)
    assert "my_enum: ColorEnum = 'red'" in text


def test_format_model_text_enum_values() -> None:
    text = format_model_text(OuterModel)
    assert "values: [red, green, blue]" in text


def test_format_model_text_constraints() -> None:
    text = format_model_text(ConstrainedModel)
    assert "constraints: ge=0.0, le=1.0" in text


def test_format_model_text_nested_expansion() -> None:
    text = format_model_text(OuterModel)
    assert "schema (InnerModel):" in text
    # Nested fields should appear indented under the schema
    assert "x: int = 0" in text
    assert "y: str = 'hello'" in text


def test_format_model_text_cycle_protection() -> None:
    text = format_model_text(SelfRefModel)
    # First level should expand
    assert "schema (SelfRefModel):" in text
    # The recursive child.child should NOT expand again (only one "schema (SelfRefModel):")
    assert text.count("schema (SelfRefModel):") == 1


def test_format_model_text_depth_limiting() -> None:
    text = format_model_text(DeepA, max_depth=1)
    # First level (DeepB) should expand
    assert "schema (DeepB):" in text


def test_format_model_text_sibling_nested_expands_each() -> None:
    """Sibling fields of the same nested type should each include a nested schema."""
    text = format_model_text(SiblingNestedModel)
    # Both first and second fields should have InnerModel expanded
    assert text.count("schema (InnerModel):") == 2


def test_format_model_text_deduplication_with_seen_schemas() -> None:
    """When seen_schemas is passed across calls, second occurrence shows a back-reference."""
    seen: set[str] = set()
    text1 = format_model_text(OuterModel, seen_schemas=seen)
    text2 = format_model_text(SiblingNestedModel, seen_schemas=seen)
    assert "schema (InnerModel):" in text1
    assert "see InnerModel above" in text2


def test_format_model_text_no_dedup_without_seen_set() -> None:
    """Without seen_schemas, nested schemas always expand fully."""
    text = format_model_text(OuterModel)
    assert "schema (InnerModel):" in text


def test_format_model_text_max_depth_zero_blocks_all_nesting() -> None:
    """At max_depth=0, nested schemas should not expand."""
    text = format_model_text(OuterModel, max_depth=0)
    assert "schema (InnerModel):" not in text
    assert "nested:" in text  # field still listed, just not expanded


def test_format_model_text_dedup_distinguishes_same_name_different_module() -> None:
    """Schemas with same __name__ but different __module__ should not dedup."""

    class SharedNameA(BaseModel):
        x: int = 0

    class SharedNameB(BaseModel):
        y: str = ""

    # Make them look like same-named classes from different modules
    SharedNameB.__name__ = "SharedNameA"
    SharedNameB.__qualname__ = "SharedNameA"
    SharedNameA.__module__ = "pkg.alpha"
    SharedNameB.__module__ = "pkg.beta"

    class WrapperA(BaseModel):
        a: SharedNameA = Field(default_factory=SharedNameA)

    class WrapperB(BaseModel):
        b: SharedNameB = Field(default_factory=SharedNameB)

    WrapperA.model_rebuild()
    WrapperB.model_rebuild()

    seen: set[str] = set()
    text_a = format_model_text(WrapperA, seen_schemas=seen)
    text_b = format_model_text(WrapperB, seen_schemas=seen)

    assert "schema (SharedNameA):" in text_a
    assert "schema (SharedNameA):" in text_b
    assert "see SharedNameA above" not in text_b


class Level3(BaseModel):
    val: int = 0


class Level2(BaseModel):
    val: int = 0
    child: Level3 | None = None


class Level1(BaseModel):
    val: int = 0
    child: Level2 | None = None


Level1.model_rebuild()
Level2.model_rebuild()


def test_format_model_text_depth_limiting_blocks_deeper_nesting() -> None:
    """With max_depth=1, Level2 expands but Level3 does not."""
    text = format_model_text(Level1, max_depth=1)
    assert "schema (Level2):" in text
    assert "schema (Level3):" not in text


# ---------------------------------------------------------------------------
# _default_to_json (P1-6)
# ---------------------------------------------------------------------------


class _JsonTestEnum(str, Enum):
    MEMBER = "member_value"


class _CustomObj:
    def __repr__(self) -> str:
        return "CustomObj()"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        (_JsonTestEnum.MEMBER, "member_value"),
        (True, True),
        (42, 42),
        (3.14, 3.14),
        ("hello", "hello"),
        ([1, 2], [1, 2]),
        ({"a": 1}, {"a": 1}),
    ],
)
def test_default_to_json(value: object, expected: object) -> None:
    assert _default_to_json(value) == expected


def test_default_to_json_custom_object() -> None:
    obj = _CustomObj()
    assert _default_to_json(obj) == "CustomObj()"


# ---------------------------------------------------------------------------
# format_type — regex branches (P1-8)
# ---------------------------------------------------------------------------


def test_format_type_none_type() -> None:
    result = format_type(type(None))
    assert result == "None"


def test_format_type_enum_class() -> None:
    result = format_type(ColorEnum)
    assert result == "ColorEnum"


def test_format_type_module_prefix_stripping() -> None:
    import data_designer.config as dd

    result = format_type(list[dd.CategorySamplerParams])
    assert "data_designer.config." not in result
    assert "CategorySamplerParams" in result


def test_format_type_literal() -> None:
    from typing import Literal

    result = format_type(Literal["foo", "bar"])
    assert "Literal[" in result
    assert "foo" in result
    assert "bar" in result


# ---------------------------------------------------------------------------
# format_model_text — empty model (P1-10)
# ---------------------------------------------------------------------------


class EmptyModel(BaseModel):
    """An empty model with no fields."""


def test_format_model_text_empty_model() -> None:
    text = format_model_text(EmptyModel)
    assert "EmptyModel:" in text
    assert "fields:" in text
    lines = text.strip().split("\n")
    field_lines = [line for line in lines if line.startswith("    ") and ":" in line and "fields:" not in line]
    assert len(field_lines) == 0
