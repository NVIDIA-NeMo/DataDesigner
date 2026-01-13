# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.processing.gsonschema.validators import JSONSchemaValidationError, validate


@pytest.fixture
def stub_ap_false_flag():
    return {"additionalProperties": False}


@pytest.fixture
def stub_simple_object_schema():
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "number"}},
        "additionalProperties": False,
    }


@pytest.fixture
def stub_simple_object_data():
    return {"name": "Alice", "age": 30, "extra": "remove me"}


@pytest.fixture
def stub_nested_object_schema():
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
                "additionalProperties": False,
            },
        },
        "additionalProperties": False,
    }


@pytest.fixture
def stub_nested_object_data():
    return {
        "name": "Bob",
        "address": {"street": "Main St", "city": "Town", "zipcode": "12345"},
        "extra": "should be removed",
    }


@pytest.fixture
def stub_array_schema():
    return {
        "type": "array",
        "items": {
            "type": "object",
            "properties": {"id": {"type": "number"}, "value": {"type": "string"}},
            "additionalProperties": False,
        },
    }


@pytest.fixture
def stub_array_data():
    return [
        {"id": 1, "value": "one", "extra": "remove"},
        {"id": 2, "value": "two", "another": "remove"},
    ]


@pytest.mark.parametrize(
    "test_case,schema_fixture,data_fixture,expected_result",
    [
        (
            "simple_object_pruning",
            "stub_simple_object_schema",
            "stub_simple_object_data",
            {"name": "Alice", "age": 30},
        ),
        (
            "nested_object_pruning",
            "stub_nested_object_schema",
            "stub_nested_object_data",
            {"name": "Bob", "address": {"street": "Main St", "city": "Town"}},
        ),
        (
            "array_pruning",
            "stub_array_schema",
            "stub_array_data",
            [{"id": 1, "value": "one"}, {"id": 2, "value": "two"}],
        ),
    ],
)
def test_pruning_scenarios(request, test_case, schema_fixture, data_fixture, expected_result):
    schema = request.getfixturevalue(schema_fixture)
    data = request.getfixturevalue(data_fixture)

    result = validate(data, schema, pruning=True)
    assert result == expected_result


@pytest.mark.parametrize(
    "test_case,schema_fixture,data_fixture,validation_method,should_raise",
    [
        ("simple_object_no_pruning", "stub_simple_object_schema", "stub_simple_object_data", "ap_false_flag", True),
        (
            "simple_object_no_extra_properties",
            "stub_simple_object_schema",
            "stub_simple_object_data",
            "no_extra_properties",
            True,
        ),
        (
            "simple_object_pruning_with_ap_false",
            "stub_simple_object_schema",
            "stub_simple_object_data",
            "pruning_with_ap_false",
            False,
        ),
        (
            "simple_object_pruning_with_no_extra",
            "stub_simple_object_schema",
            "stub_simple_object_data",
            "pruning_with_no_extra",
            False,
        ),
    ],
)
def test_validation_scenarios(request, test_case, schema_fixture, data_fixture, validation_method, should_raise):
    schema = request.getfixturevalue(schema_fixture)
    data = request.getfixturevalue(data_fixture)

    if validation_method == "ap_false_flag":
        ap_false_flag = request.getfixturevalue("stub_ap_false_flag")
        if should_raise:
            with pytest.raises(JSONSchemaValidationError):
                validate(data, schema | ap_false_flag)
        else:
            result = validate(data, schema | ap_false_flag)
            assert result == data
    elif validation_method == "no_extra_properties":
        if should_raise:
            with pytest.raises(JSONSchemaValidationError):
                validate(data, schema, no_extra_properties=True)
        else:
            result = validate(data, schema, no_extra_properties=True)
            assert result == data
    elif validation_method == "pruning_with_ap_false":
        ap_false_flag = request.getfixturevalue("stub_ap_false_flag")
        result = validate(data, schema | ap_false_flag, pruning=True)
        assert "extra" not in result
        assert result == {"name": "Alice", "age": 30}
    elif validation_method == "pruning_with_no_extra":
        result = validate(data, schema, pruning=True, no_extra_properties=True)
        assert "extra" not in result
        assert result == {"name": "Alice", "age": 30}


@pytest.mark.parametrize(
    "test_case,schema,data,expected_result",
    [
        (
            "no_extra_properties_no_changes",
            {
                "type": "object",
                "properties": {"foo": {"type": "string"}},
                "additionalProperties": False,
            },
            {"foo": "bar"},
            {"foo": "bar"},
        ),
        (
            "non_dict_instance",
            {"type": "string"},
            "just a string",
            "just a string",
        ),
    ],
)
def test_special_cases(test_case, schema, data, expected_result):
    result = validate(data, schema, pruning=True)
    assert result == expected_result


def test_invalid_data_type():
    schema = {
        "type": "object",
        "properties": {"num": {"type": "number"}},
    }

    data = {"num": "not a number", "extra": "should be removed"}
    with pytest.raises(JSONSchemaValidationError):
        validate(data, schema, pruning=True, no_extra_properties=True)


def test_normalize_decimal_anyof_fields() -> None:
    """Test that Decimal-like anyOf fields are normalized to floats with proper precision."""
    schema = {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "price": {
                "anyOf": [
                    {"type": "number"},
                    {"type": "string", "pattern": r"^(?!^[-+.]*$)[+-]?0*\d*\.?\d{0,2}0*$"},
                ]
            },
        },
    }

    # Numeric value with extra precision should be rounded to 2 decimal places
    result1 = validate({"name": "Widget", "price": 189.999}, schema)
    assert result1["price"] == 190.0
    assert isinstance(result1["price"], float)

    # Numeric value should be converted to float
    result2 = validate({"name": "Gadget", "price": 50.5}, schema)
    assert result2["price"] == 50.5
    assert isinstance(result2["price"], float)

    # String value should be converted to float
    result3 = validate({"name": "Gizmo", "price": "249.99"}, schema)
    assert result3["price"] == 249.99
    assert isinstance(result3["price"], float)
