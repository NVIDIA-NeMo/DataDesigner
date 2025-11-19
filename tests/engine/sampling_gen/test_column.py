# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pydantic import ValidationError
import pytest

from data_designer.config.sampler_params import PersonSamplerParams, UUIDSamplerParams
from data_designer.engine.sampling_gen.column import ConditionalDataColumn


def test_init(stub_default_samplers):
    params_list = stub_default_samplers["params"]
    sampler_type_list = stub_default_samplers["sampler_types"]
    for sampler_type, params in zip(sampler_type_list, params_list, strict=True):
        column = ConditionalDataColumn(name="testing", sampler_type=sampler_type, params=params)
        assert column.name == "testing"
        assert column.sampler_type == sampler_type
        assert column.conditional_params == {}
        assert column.conditions == ["..."]
        # Check that the dumped params contains all the original params (may have additional fields with default values)
        dumped_params = column.params.model_dump(mode="json")
        for key, value in params.items():
            assert key in dumped_params
            assert dumped_params[key] == value


def test_conditional_params():
    column = ConditionalDataColumn(
        name="column",
        sampler_type="gaussian",
        params={"mean": 0.0, "stddev": 1.0},
        conditional_params={
            "col_2 == 'this_value'": {"mean": 0.0, "stddev": 1.0},
            "col_3 == 'not_this_value'": {"mean": 1.0, "stddev": 2.0},
        },
    )
    assert column.conditional_column_names == {"col_2", "col_3"}
    assert column.conditions == [
        "col_2 == 'this_value'",
        "col_3 == 'not_this_value'",
        "not ((col_2 == 'this_value') or (col_3 == 'not_this_value'))",
    ]


@pytest.mark.parametrize(
    "test_case,sampler_type,params,expected_params_class",
    [
        ("uuid_serialization", "uuid", {}, UUIDSamplerParams),
        ("person_serialization", "person", {}, PersonSamplerParams),
    ],
)
def test_default_samplers_can_serialize(test_case, sampler_type, params, expected_params_class):
    try:
        column = ConditionalDataColumn(name="column", sampler_type=sampler_type, params=params)
        column.model_dump(exclude_unset=True)
    except Exception as e:
        pytest.fail(f"Serialization should succeed: {e}")

    assert isinstance(column.params, expected_params_class)


@pytest.mark.parametrize(
    "test_case,sampler_type,params,expected_error",
    [
        ("no_sampler_type", None, {"mean": 0.0, "stddev": 1.0}, ValidationError),
        ("invalid_sampler_type", "no_a_type", {}, ValidationError),
    ],
)
def test_validation_error_cases(test_case, sampler_type, params, expected_error):
    if test_case == "no_sampler_type":
        with pytest.raises(expected_error):
            ConditionalDataColumn(name="testing", params=params)
    elif test_case == "invalid_sampler_type":
        with pytest.raises(
            expected_error,
            match="Invalid sampler type: no_a_type. Available samplers: ",
        ):
            ConditionalDataColumn(name="testing", sampler_type=sampler_type, params=params)


@pytest.mark.parametrize(
    "test_case,sampler_type,params,convert_to,expected_error",
    [
        ("invalid_convert_to_gaussian", "gaussian", {"mean": 0.0, "stddev": 1.0}, "invalid", ValidationError),
        (
            "invalid_convert_to_datetime",
            "datetime",
            {"start": "2021-01-01", "end": "2021-01-02"},
            "not/a/valid/format",
            ValidationError,
        ),
    ],
)
def test_invalid_convert_to_scenarios(test_case, sampler_type, params, convert_to, expected_error):
    with pytest.raises(expected_error):
        ConditionalDataColumn(
            name="testing",
            sampler_type=sampler_type,
            params=params,
            convert_to=convert_to,
        )
