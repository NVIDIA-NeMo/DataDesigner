# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum

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


class MixedModel(ConfigBase):
    """Model exercising every field variant."""

    required_field: str
    optional_none: int | None = None
    with_default: float = 3.14
    enum_field: Color = Color.GREEN
    factory_list: list[str] = Field(default_factory=list)
    factory_dict: dict[str, int] = Field(default_factory=dict, description="A mapping")


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
    name_idx = next(i for i, l in enumerate(lines) if "name: str" in l)
    assert "The name of the thing" in lines[name_idx + 1]


# --- Docstrings ---


def test_docstring_included() -> None:
    text = DocstringModel.schema_text()
    assert "A model with a docstring summary." in text


def test_no_docstring_still_works() -> None:
    text = NoDocstringModel.schema_text()
    assert text.startswith("NoDocstringModel:")
    assert "value: int  [required]" in text


# --- Header format ---


def test_header_is_class_name() -> None:
    text = RequiredOnlyModel.schema_text()
    assert text.startswith("RequiredOnlyModel:")


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
