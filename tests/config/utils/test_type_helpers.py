# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Literal, Union

from pydantic import BaseModel
import pytest

from data_designer.config.utils.errors import (
    InvalidDiscriminatorFieldError,
    InvalidEnumValueError,
    InvalidTypeUnionError,
)
from data_designer.config.utils.type_helpers import (
    SAMPLER_PARAMS,
    create_str_enum_from_discriminated_type_union,
    get_sampler_params,
    resolve_string_enum,
)


class StubTestEnum(str, Enum):
    TEST = "test"


class StubModelA(BaseModel):
    column_type: Literal["type-a", "type-a-alt"] = "type-a"
    name: str


class StubModelB(BaseModel):
    column_type: Literal["type-b"] = "type-b"
    value: int


class StubModelC(BaseModel):
    column_type: Literal["type-c-with-dashes"] = "type-c-with-dashes"
    data: str


class StubModelWithoutDiscriminator(BaseModel):
    name: str
    value: int


class NotAModel:
    column_type: str = "not-a-model"


def test_create_str_enum_from_type_union_basic() -> None:
    type_union = Union[StubModelA, StubModelB]
    result = create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")

    assert issubclass(result, Enum)
    assert issubclass(result, str)
    assert hasattr(result, "TYPE_A")
    assert hasattr(result, "TYPE_A_ALT")
    assert hasattr(result, "TYPE_B")
    assert result.TYPE_A.value == "type-a"
    assert result.TYPE_A_ALT.value == "type-a-alt"
    assert result.TYPE_B.value == "type-b"
    assert len(result) == 3


def test_create_str_enum_from_type_union_with_dashes() -> None:
    type_union = Union[StubModelC, StubModelA]
    result = create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")

    assert hasattr(result, "TYPE_C_WITH_DASHES")
    assert result.TYPE_C_WITH_DASHES.value == "type-c-with-dashes"


def test_create_str_enum_from_type_union_multiple_models() -> None:
    type_union = Union[StubModelA, StubModelB, StubModelC]
    result = create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")

    assert len(result) == 4
    assert hasattr(result, "TYPE_A")
    assert hasattr(result, "TYPE_A_ALT")
    assert hasattr(result, "TYPE_B")
    assert hasattr(result, "TYPE_C_WITH_DASHES")


def test_create_str_enum_from_type_union_duplicate_values() -> None:
    class StubModelD(BaseModel):
        column_type: Literal["type-a"] = "type-a"
        extra: str

    type_union = Union[StubModelA, StubModelD]
    result = create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")

    assert len(result) == 2
    assert hasattr(result, "TYPE_A")
    assert hasattr(result, "TYPE_A_ALT")


def test_create_str_enum_from_type_union_not_pydantic_model() -> None:
    type_union = Union[StubModelA, NotAModel]

    with pytest.raises(InvalidTypeUnionError, match="must be a subclass of pydantic.BaseModel"):
        create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")


def test_create_str_enum_from_type_union_invalid_discriminator_field() -> None:
    type_union = Union[StubModelA, StubModelWithoutDiscriminator]

    with pytest.raises(InvalidDiscriminatorFieldError, match="'column_type' is not a field of"):
        create_str_enum_from_discriminated_type_union("TestEnum", type_union, "column_type")

    with pytest.raises(InvalidDiscriminatorFieldError, match="'name' must be a Literal type"):
        create_str_enum_from_discriminated_type_union("TestEnum", type_union, "name")


def test_create_str_enum_from_type_union_custom_discriminator_name() -> None:
    class StubModelE(BaseModel):
        type_field: Literal["custom-type"] = "custom-type"
        name: str

    class StubModelF(BaseModel):
        type_field: Literal["another-type"] = "another-type"
        value: int

    type_union = Union[StubModelE, StubModelF]
    result = create_str_enum_from_discriminated_type_union("TestEnum", type_union, "type_field")

    assert hasattr(result, "CUSTOM_TYPE")
    assert result.CUSTOM_TYPE.value == "custom-type"
    assert hasattr(result, "ANOTHER_TYPE")
    assert result.ANOTHER_TYPE.value == "another-type"


def test_get_sampler_params():
    expected_sampler_keys = {
        "bernoulli",
        "bernoulli_mixture",
        "binomial",
        "category",
        "datetime",
        "gaussian",
        "person",
        "person_from_faker",
        "poisson",
        "scipy",
        "subcategory",
        "timedelta",
        "uniform",
        "uuid",
    }
    assert set(get_sampler_params().keys()) == expected_sampler_keys
    assert set(SAMPLER_PARAMS.keys()) == expected_sampler_keys


def test_resolve_string_enum():
    with pytest.raises(InvalidEnumValueError, match="`enum_type` must be a subclass of Enum"):
        resolve_string_enum("invalid", int)
    assert resolve_string_enum(StubTestEnum.TEST, StubTestEnum) == StubTestEnum.TEST
    assert resolve_string_enum("test", StubTestEnum) == StubTestEnum.TEST
    with pytest.raises(InvalidEnumValueError, match="'invalid' is not a valid string enum"):
        resolve_string_enum("invalid", StubTestEnum)
    with pytest.raises(InvalidEnumValueError, match="'1' is not a valid string enum"):
        resolve_string_enum(1, StubTestEnum)
