# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import Field

from data_designer.config.base import ConfigBase


class Color(Enum):
    RED = "red"
    GREEN = "green"


class RequiredOnlyModel(ConfigBase):
    name: str
    count: int


class DefaultFactoryModel(ConfigBase):
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, int] = Field(default_factory=dict)


class DefaultFactoryWithDescriptionModel(ConfigBase):
    items: list[str] = Field(default_factory=list, description="Collection of items")


class EnumDefaultModel(ConfigBase):
    color: Color = Color.RED


class NoneDefaultModel(ConfigBase):
    label: str | None = None


class ScalarDefaultModel(ConfigBase):
    threshold: float = 0.5
    enabled: bool = True
    tag: str = "default"


class DescribedFieldsModel(ConfigBase):
    name: str = Field(description="The name of the thing")
    count: int = Field(default=0, description="How many things")


class DocstringModel(ConfigBase):
    """A model with a docstring summary."""

    value: int


class NoDocstringModel(ConfigBase):
    value: int


class LiteralModel(ConfigBase):
    tag: Literal["fixed"] = "fixed"
    name: str


class MixedModel(ConfigBase):
    """Model exercising every field variant."""

    required_field: str
    optional_none: int | None = None
    with_default: float = 3.14
    enum_field: Color = Color.GREEN
    factory_list: list[str] = Field(default_factory=list)
    factory_dict: dict[str, int] = Field(default_factory=dict, description="A mapping")


class NestedChild(ConfigBase):
    """A child model for nesting tests."""

    x: str
    y: int = 0


class ParentWithNested(ConfigBase):
    """Parent model with a nested ConfigBase field."""

    label: str
    child: NestedChild


class ParentWithNestedList(ConfigBase):
    """Parent with a list of nested ConfigBase."""

    items: list[NestedChild]


class ParentWithOptionalNested(ConfigBase):
    """Parent with an optional nested ConfigBase."""

    child: NestedChild | None = None


class EnumFieldModel(ConfigBase):
    color: Color = Field(description="Pick a color")


class ReprFalseModel(ConfigBase):
    visible: str
    hidden: bool = Field(default=False, repr=False)


class DiscriminatorModel(ConfigBase):
    column_type: Literal["test"] = "test"
    name: str


class MultiRequiredModel(ConfigBase):
    """A model with multiple required fields."""

    name: str
    count: int
    label: str


class NoRequiredModel(ConfigBase):
    """A model with no required fields."""

    x: int = 0
    y: str = "hi"


class MultiLiteralModel(ConfigBase):
    mode: Literal["a", "b"] = "a"
    name: str


class RequiredEnumModel(ConfigBase):
    color: Color


class DocstringWithSectionModel(ConfigBase):
    """Summary paragraph.

    Attributes:
        value: some description
    """

    value: int


class AllHiddenModel(ConfigBase):
    """All fields are hidden."""

    discriminator: Literal["test"] = "test"
    internal: bool = Field(default=False, repr=False)


# --- Required fields ---


def test_required_fields_marked() -> None:
    text = RequiredOnlyModel.schema_text()
    assert "name: str  [required]" in text
    assert "count: int  [required]" in text


# --- default_factory fields ---


def test_default_factory_list_shows_factory_call() -> None:
    text = DefaultFactoryModel.schema_text()
    assert "tags:" in text
    assert "= list()" in text


def test_default_factory_dict_shows_factory_call() -> None:
    text = DefaultFactoryModel.schema_text()
    assert "metadata:" in text
    assert "= dict()" in text


def test_default_factory_does_not_show_pydantic_undefined() -> None:
    text = DefaultFactoryModel.schema_text()
    assert "PydanticUndefined" not in text


def test_default_factory_with_description() -> None:
    text = DefaultFactoryWithDescriptionModel.schema_text()
    assert "items:" in text
    assert "= list()" in text
    assert "Collection of items" in text


# --- Enum defaults ---


def test_enum_default_shows_value() -> None:
    text = EnumDefaultModel.schema_text()
    assert "color: Color = 'red'" in text
    assert "Color.RED" not in text


# --- None defaults ---


def test_none_default() -> None:
    text = NoneDefaultModel.schema_text()
    assert "label: str | None = None" in text


# --- Scalar defaults ---


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("threshold", "threshold: float = 0.5"),
        ("enabled", "enabled: bool = True"),
        ("tag", "tag: str = 'default'"),
    ],
    ids=["float", "bool", "str"],
)
def test_scalar_defaults(field_name: str, expected: str) -> None:
    text = ScalarDefaultModel.schema_text()
    assert expected in text


# --- Field descriptions ---


def test_description_appears_below_field() -> None:
    text = DescribedFieldsModel.schema_text()
    lines = text.splitlines()
    name_idx = next(i for i, line in enumerate(lines) if "name: str" in line)
    assert "The name of the thing" in lines[name_idx + 1]


# --- Docstrings ---


def test_docstring_included() -> None:
    text = DocstringModel.schema_text()
    assert "A model with a docstring summary." in text


def test_no_docstring_still_works() -> None:
    text = NoDocstringModel.schema_text()
    assert text.startswith("NoDocstringModel:")
    assert "value: int  [required]" in text


def test_docstring_truncated_at_section_header() -> None:
    text = DocstringWithSectionModel.schema_text()
    assert "Summary paragraph." in text
    assert "Attributes:" not in text


# --- Header format ---


def test_header_is_class_name() -> None:
    text = RequiredOnlyModel.schema_text()
    assert text.startswith("RequiredOnlyModel:")


# --- Discriminator suppression ---


def test_discriminator_field_is_suppressed() -> None:
    text = DiscriminatorModel.schema_text()
    assert "column_type" not in text
    assert "name: str  [required]" in text


def test_literal_field_with_matching_default_is_discriminator() -> None:
    text = LiteralModel.schema_text()
    assert "tag:" not in text
    assert "name: str  [required]" in text


def test_multi_value_literal_is_not_discriminator() -> None:
    text = MultiLiteralModel.schema_text()
    assert "mode:" in text


# --- repr=False suppression ---


def test_repr_false_field_is_suppressed() -> None:
    text = ReprFalseModel.schema_text()
    assert "visible: str  [required]" in text
    assert "hidden" not in text


# --- Enum values display ---


def test_enum_values_shown_after_description() -> None:
    text = EnumFieldModel.schema_text()
    assert "Pick a color" in text
    assert "values: red, green" in text


def test_enum_default_field_shows_values() -> None:
    text = EnumDefaultModel.schema_text()
    assert "values: red, green" in text


def test_required_enum_field_shows_values() -> None:
    text = RequiredEnumModel.schema_text()
    assert "color: Color  [required]" in text
    assert "values: red, green" in text


# --- Nested ConfigBase expansion ---


def test_nested_configbase_expanded() -> None:
    text = ParentWithNested.schema_text()
    assert "NestedChild:" in text
    assert "A child model for nesting tests." in text
    assert "x: str  [required]" in text


def test_nested_list_configbase_expanded() -> None:
    text = ParentWithNestedList.schema_text()
    assert "items: list[NestedChild]  [required]" in text
    assert "NestedChild:" in text


def test_nested_optional_configbase_expanded() -> None:
    text = ParentWithOptionalNested.schema_text()
    assert "NestedChild:" in text


def test_inherited_generic_field_expanded() -> None:
    """Inherited list[ConfigBase] fields must resolve generics and expand, even on Python 3.10."""

    class Inner(ConfigBase):
        val: str

    class Parent(ConfigBase):
        items: list[Inner]

    class Child(Parent):
        extra: int = 0

    text = Child.schema_text()
    assert "items: list[Inner]  [required]" in text
    assert "Inner:" in text
    assert "val: str  [required]" in text


def test_multi_member_union_not_expanded() -> None:
    class TypeA(ConfigBase):
        a: str

    class TypeB(ConfigBase):
        b: str

    class ModelWithUnion(ConfigBase):
        child: TypeA | TypeB

    text = ModelWithUnion.schema_text()
    assert "TypeA:" not in text
    assert "TypeB:" not in text


def test_nested_expansion_limited_to_one_level() -> None:
    """Nested models at depth >= 1 should not expand their own nested fields."""

    class GrandChild(ConfigBase):
        """Deep model."""

        val: str

    class MiddleChild(ConfigBase):
        """Middle model."""

        gc: GrandChild

    class TopLevel(ConfigBase):
        """Top model."""

        mc: MiddleChild

    text = TopLevel.schema_text()
    assert "MiddleChild:" in text
    assert "GrandChild:" not in text


def test_nested_model_has_no_example_line() -> None:
    text = ParentWithNested.schema_text()
    assert "Example: dd.NestedChild(" not in text


# --- Instantiation example ---


def test_instantiation_example_at_depth_zero() -> None:
    text = MultiRequiredModel.schema_text()
    assert "Example: dd.MultiRequiredModel(name=..., count=..., label=...)" in text


def test_no_example_when_no_required_fields() -> None:
    text = NoRequiredModel.schema_text()
    assert "Example:" not in text


def test_example_excludes_discriminator_fields() -> None:
    text = DiscriminatorModel.schema_text()
    assert "Example: dd.DiscriminatorModel(name=...)" in text
    assert "column_type" not in text


# --- All fields hidden ---


def test_all_fields_hidden_produces_valid_output() -> None:
    text = AllHiddenModel.schema_text()
    assert "AllHiddenModel:" in text
    assert "All fields are hidden." in text
    assert "discriminator" not in text
    assert "internal" not in text
    assert "Example:" not in text


# --- Mixed model exercises all variants together ---


def test_mixed_model_all_variants() -> None:
    text = MixedModel.schema_text()
    assert "Model exercising every field variant." in text
    assert "required_field: str  [required]" in text
    assert "optional_none: int | None = None" in text
    assert "with_default: float = 3.14" in text
    assert "enum_field: Color = 'green'" in text
    assert "factory_list:" in text
    assert "= list()" in text
    assert "factory_dict:" in text
    assert "= dict()" in text
    assert "A mapping" in text
    assert "PydanticUndefined" not in text
    assert "values: red, green" in text
    assert "Example: dd.MixedModel(required_field=...)" in text
