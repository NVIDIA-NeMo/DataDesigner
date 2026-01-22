# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any

import pytest

from data_designer.engine.processing.gsonschema.types import (
    DataObjectT,
    JSONSchemaT,
    T_primitive,
)


@pytest.fixture
def stub_valid_data_objects():
    return [
        {"key": "value"},  # dict
        [1, 2, 3],  # list
        "string",  # str
        42,  # int
        3.14,  # float
        True,  # bool
    ]


@pytest.fixture
def stub_valid_primitives():
    return [
        "string",  # str
        42,  # int
        3.14,  # float
        True,  # bool
    ]


@pytest.fixture
def stub_sample_schema():
    return {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}}


@pytest.mark.parametrize(
    "test_case,test_data_fixture,expected_types",
    [
        ("data_object_t_union", "stub_valid_data_objects", (dict, list, str, int, float, bool)),
        ("t_primitive_constraint", "stub_valid_primitives", (str, int, float, bool)),
    ],
)
def test_type_validations(request, test_case, test_data_fixture, expected_types):
    test_data = request.getfixturevalue(test_data_fixture)

    for obj in test_data:
        assert isinstance(obj, expected_types)


@pytest.mark.parametrize(
    "test_case,test_value,expected_type",
    [
        ("json_schema_t_type", "stub_sample_schema", dict),
        ("json_schema_t_keys", "stub_sample_schema", str),
    ],
)
def test_json_schema_scenarios(request, test_case, test_value, expected_type):
    schema = request.getfixturevalue(test_value)

    if test_case == "json_schema_t_type":
        assert isinstance(schema, expected_type)
    elif test_case == "json_schema_t_keys":
        assert all(isinstance(key, expected_type) for key in schema.keys())


@pytest.mark.parametrize(
    "test_case,type_name,expected_definition",
    [
        ("type_constraints", "DataObjectT", (dict | list | str | int | float | bool)),
        ("type_constraints", "JSONSchemaT", dict[str, Any]),
        ("t_primitive_constraints", "T_primitive", (str, int, float, bool)),
    ],
)
def test_type_definitions_and_constraints(test_case, type_name, expected_definition):
    type_mapping = {
        "DataObjectT": DataObjectT,
        "JSONSchemaT": JSONSchemaT,
        "T_primitive": T_primitive,
    }

    if test_case == "type_constraints":
        assert type_mapping[type_name] == expected_definition
    elif test_case == "t_primitive_constraints":
        assert hasattr(T_primitive, "__constraints__")
        constraints = T_primitive.__constraints__
        for constraint_type in expected_definition:
            assert constraint_type in constraints


@pytest.mark.parametrize(
    "type_name,test_value,expected_type",
    [
        ("DataObjectT", {"name": "John", "age": 30}, dict),
        (
            "JSONSchemaT",
            {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            dict,
        ),
        ("T_primitive", "hello", str),
    ],
)
def test_type_usage_examples(type_name, test_value, expected_type):
    assert isinstance(test_value, expected_type)
