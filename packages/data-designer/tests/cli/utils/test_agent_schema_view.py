# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Literal

import pytest
from pydantic import Field

from data_designer.cli.utils.agent_schema_view import (
    PydanticFieldView,
    PydanticModelView,
    describe_pydantic_model,
)
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


class AgentHiddenModel(ConfigBase):
    visible: str
    hidden: bool = False


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


class RequiredLiteralModel(ConfigBase):
    tag: Literal["only"]
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
    internal: bool = False


def get_field(view: PydanticModelView, field_name: str) -> PydanticFieldView:
    return next(field for field in view.fields if field.name == field_name)


def test_required_fields_marked() -> None:
    view = describe_pydantic_model(RequiredOnlyModel)

    assert view.required_field_names == ("name", "count")
    assert get_field(view, "name").required is True
    assert get_field(view, "count").required is True


def test_default_factory_list_shows_factory_call() -> None:
    view = describe_pydantic_model(DefaultFactoryModel)
    assert get_field(view, "tags").default_text == "list()"


def test_default_factory_dict_shows_factory_call() -> None:
    view = describe_pydantic_model(DefaultFactoryModel)
    assert get_field(view, "metadata").default_text == "dict()"


def test_default_factory_does_not_show_pydantic_undefined() -> None:
    view = describe_pydantic_model(DefaultFactoryModel)
    assert all("PydanticUndefined" not in (field.default_text or "") for field in view.fields)


def test_default_factory_with_description() -> None:
    view = describe_pydantic_model(DefaultFactoryWithDescriptionModel)
    field = get_field(view, "items")

    assert field.default_text == "list()"
    assert field.description == "Collection of items"


def test_enum_default_shows_value() -> None:
    view = describe_pydantic_model(EnumDefaultModel)
    field = get_field(view, "color")

    assert field.type_text == "Color"
    assert field.default_text == "'red'"
    assert field.enum_values == ("red", "green")


def test_none_default() -> None:
    view = describe_pydantic_model(NoneDefaultModel)
    field = get_field(view, "label")

    assert field.type_text == "str | None"
    assert field.default_text == "None"


@pytest.mark.parametrize(
    ("field_name", "expected"),
    [
        ("threshold", "0.5"),
        ("enabled", "True"),
        ("tag", "'default'"),
    ],
    ids=["float", "bool", "str"],
)
def test_scalar_defaults(field_name: str, expected: str) -> None:
    view = describe_pydantic_model(ScalarDefaultModel)
    assert get_field(view, field_name).default_text == expected


def test_description_appears_on_field_view() -> None:
    view = describe_pydantic_model(DescribedFieldsModel)
    assert get_field(view, "name").description == "The name of the thing"


def test_docstring_included() -> None:
    view = describe_pydantic_model(DocstringModel)
    assert view.summary == "A model with a docstring summary."


def test_no_docstring_still_works() -> None:
    view = describe_pydantic_model(NoDocstringModel)
    assert view.class_name == "NoDocstringModel"
    assert get_field(view, "value").required is True


def test_docstring_truncated_at_section_header() -> None:
    view = describe_pydantic_model(DocstringWithSectionModel)
    assert view.summary == "Summary paragraph."


def test_header_is_class_name() -> None:
    view = describe_pydantic_model(RequiredOnlyModel)
    assert view.class_name == "RequiredOnlyModel"


def test_hidden_fields_suppress_discriminator_field() -> None:
    view = describe_pydantic_model(DiscriminatorModel, hidden_fields={"column_type"})
    assert [field.name for field in view.fields] == ["name"]


def test_literal_field_with_matching_default_is_visible_without_hidden_fields() -> None:
    view = describe_pydantic_model(LiteralModel)
    assert [field.name for field in view.fields] == ["tag", "name"]


def test_multi_value_literal_is_not_suppressed() -> None:
    view = describe_pydantic_model(MultiLiteralModel)
    assert get_field(view, "mode").type_text == "Literal['a', 'b']"


def test_required_single_literal_is_not_suppressed() -> None:
    view = describe_pydantic_model(RequiredLiteralModel)
    assert get_field(view, "tag").required is True


def test_hidden_fields_parameter_suppresses_named_fields() -> None:
    view = describe_pydantic_model(AgentHiddenModel, hidden_fields={"hidden"})
    assert [field.name for field in view.fields] == ["visible"]


def test_enum_values_shown_on_field_view() -> None:
    view = describe_pydantic_model(EnumFieldModel)
    field = get_field(view, "color")

    assert field.description == "Pick a color"
    assert field.enum_values == ("red", "green")


def test_required_enum_field_shows_values() -> None:
    view = describe_pydantic_model(RequiredEnumModel)
    field = get_field(view, "color")

    assert field.required is True
    assert field.enum_values == ("red", "green")


def test_nested_configbase_expanded() -> None:
    view = describe_pydantic_model(ParentWithNested)
    field = get_field(view, "child")
    nested = field.nested_models[0]

    assert nested.class_name == "NestedChild"
    assert nested.summary == "A child model for nesting tests."
    assert get_field(nested, "x").required is True


def test_nested_list_configbase_expanded() -> None:
    view = describe_pydantic_model(ParentWithNestedList)
    field = get_field(view, "items")

    assert field.type_text == "list[NestedChild]"
    assert field.nested_models[0].class_name == "NestedChild"


def test_nested_optional_configbase_expanded() -> None:
    view = describe_pydantic_model(ParentWithOptionalNested)
    assert get_field(view, "child").nested_models[0].class_name == "NestedChild"


def test_inherited_generic_field_expanded() -> None:
    class Inner(ConfigBase):
        val: str

    class Parent(ConfigBase):
        items: list[Inner]

    class Child(Parent):
        extra: int = 0

    view = describe_pydantic_model(Child)
    field = get_field(view, "items")

    assert field.type_text == "list[Inner]"
    assert field.nested_models[0].class_name == "Inner"
    assert get_field(field.nested_models[0], "val").required is True


def test_multi_member_union_not_expanded() -> None:
    class TypeA(ConfigBase):
        a: str

    class TypeB(ConfigBase):
        b: str

    class ModelWithUnion(ConfigBase):
        child: TypeA | TypeB

    view = describe_pydantic_model(ModelWithUnion)
    assert get_field(view, "child").nested_models == ()


def test_nested_expansion_limited_to_one_level() -> None:
    class GrandChild(ConfigBase):
        """Deep model."""

        val: str

    class MiddleChild(ConfigBase):
        """Middle model."""

        gc: GrandChild

    class TopLevel(ConfigBase):
        """Top model."""

        mc: MiddleChild

    view = describe_pydantic_model(TopLevel)
    nested = get_field(view, "mc").nested_models[0]

    assert nested.class_name == "MiddleChild"
    assert get_field(nested, "gc").nested_models == ()


def test_duplicate_nested_type_expanded_once() -> None:
    class Shared(ConfigBase):
        """Shared model."""

        val: str

    class Host(ConfigBase):
        first: Shared
        second: Shared

    view = describe_pydantic_model(Host)

    assert len(get_field(view, "first").nested_models) == 1
    assert get_field(view, "second").nested_models == ()


def test_no_required_fields_keeps_required_names_empty() -> None:
    view = describe_pydantic_model(NoRequiredModel)
    assert view.required_field_names == ()


def test_hidden_required_fields_are_removed_from_required_names() -> None:
    view = describe_pydantic_model(DiscriminatorModel, hidden_fields={"column_type"})
    assert view.required_field_names == ("name",)


def test_all_fields_hidden_produces_valid_model_view() -> None:
    view = describe_pydantic_model(AllHiddenModel, hidden_fields={"discriminator", "internal"})

    assert view.class_name == "AllHiddenModel"
    assert view.summary == "All fields are hidden."
    assert view.fields == ()
    assert view.required_field_names == ()


def test_mixed_model_all_variants() -> None:
    view = describe_pydantic_model(MixedModel)

    assert view.summary == "Model exercising every field variant."
    assert get_field(view, "required_field").required is True
    assert get_field(view, "optional_none").type_text == "int | None"
    assert get_field(view, "optional_none").default_text == "None"
    assert get_field(view, "with_default").default_text == "3.14"
    assert get_field(view, "enum_field").default_text == "'green'"
    assert get_field(view, "enum_field").enum_values == ("red", "green")
    assert get_field(view, "factory_list").default_text == "list()"
    assert get_field(view, "factory_dict").default_text == "dict()"
    assert get_field(view, "factory_dict").description == "A mapping"
