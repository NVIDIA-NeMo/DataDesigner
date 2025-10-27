# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum

import pytest

from data_designer.config.utils.errors import InvalidEnumValueError
from data_designer.config.utils.type_helpers import SAMPLER_PARAMS, get_sampler_params, resolve_string_enum


class StubTestEnum(str, Enum):
    TEST = "test"


def test_get_sampler_params():
    expected_sampler_keys = {
        "bernoulli",
        "bernoulli_mixture",
        "binomial",
        "category",
        "datetime",
        "gaussian",
        "person",
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
