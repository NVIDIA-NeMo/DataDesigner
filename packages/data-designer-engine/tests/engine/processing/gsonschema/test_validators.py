# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

import data_designer.lazy_heavy_imports as lazy
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


DISCRIMINATED_UNION_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {
            "type": "array",
            "items": {
                "oneOf": [{"$ref": "#/$defs/AlphaItem"}, {"$ref": "#/$defs/BetaItem"}],
                "discriminator": {
                    "propertyName": "kind",
                    "mapping": {"alpha": "#/$defs/AlphaItem", "beta": "#/$defs/BetaItem"},
                },
            },
        },
    },
    "$defs": {
        "AlphaItem": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "const": "alpha"},
                "name": {"type": "string"},
                "alpha_detail": {"type": "string"},
            },
            "required": ["kind", "name", "alpha_detail"],
        },
        "BetaItem": {
            "type": "object",
            "properties": {
                "kind": {"type": "string", "const": "beta"},
                "name": {"type": "string"},
                "beta_tags": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["kind", "name", "beta_tags"],
        },
    },
}


@pytest.mark.parametrize(
    "item,expected_keys",
    [
        ({"kind": "alpha", "name": "A", "alpha_detail": "d", "beta_tags": ["leak"]}, {"kind", "name", "alpha_detail"}),
        ({"kind": "beta", "name": "B", "beta_tags": ["t"], "alpha_detail": "leak"}, {"kind", "name", "beta_tags"}),
    ],
    ids=["alpha_with_leaked_beta_field", "beta_with_leaked_alpha_field"],
)
def test_discriminated_union_prunes_leaked_properties(item: dict, expected_keys: set) -> None:
    data = {"items": [item]}
    result = validate(data, DISCRIMINATED_UNION_SCHEMA, pruning=True, no_extra_properties=True)
    assert set(result["items"][0].keys()) == expected_keys


def test_discriminated_union_invalid_discriminator_value() -> None:
    data = {"items": [{"kind": "gamma", "name": "G"}]}
    with pytest.raises(JSONSchemaValidationError):
        validate(data, DISCRIMINATED_UNION_SCHEMA, pruning=True, no_extra_properties=True)


def test_non_discriminated_one_of_fallback() -> None:
    schema = {
        "type": "object",
        "properties": {
            "value": {
                "oneOf": [
                    {"type": "string"},
                    {"type": "number"},
                ],
            },
        },
    }
    assert validate({"value": "hello"}, schema, pruning=True)["value"] == "hello"
    assert validate({"value": 42}, schema, pruning=True)["value"] == 42
    with pytest.raises(JSONSchemaValidationError):
        validate({"value": []}, schema, pruning=True)


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


@pytest.mark.parametrize(
    "schema,data",
    [
        ({"type": "array", "items": True}, [1, "two"]),
        (
            {"type": "object", "properties": {"metadata": True}},
            {"metadata": {"score": 9}},
        ),
    ],
    ids=["boolean_items", "boolean_property"],
)
def test_boolean_subschema_does_not_crash_numeric_normalization(schema: dict, data: dict | list) -> None:
    assert validate(data, schema) == data


def test_boolean_anyof_schema_does_not_crash_decimal_detection() -> None:
    schema = {"anyOf": [True, {"type": "number"}]}

    assert validate(1, schema) == 1


@pytest.mark.parametrize("keyword", ["oneOf", "anyOf"])
def test_composition_normalization_includes_sibling_properties(keyword: str) -> None:
    schema = {
        "type": "object",
        keyword: [{"required": ["a"]}, {"required": ["b"]}],
        "properties": {
            "a": {"type": "integer"},
            "score": {"type": "number"},
        },
    }

    result = validate({"a": 1, "score": 9}, schema)

    assert result["a"] == 1
    assert isinstance(result["a"], int)
    assert result["score"] == 9.0
    assert isinstance(result["score"], float)


@pytest.mark.parametrize(
    "value,expected_type",
    [(9, float), (None, type(None))],
    ids=["number", "null"],
)
def test_normalize_nullable_number(value, expected_type: type) -> None:
    schema = {"anyOf": [{"type": "number"}, {"type": "null"}]}

    result = validate(value, schema)

    assert isinstance(result, expected_type)


NESTED_EVALUATION_SCHEMA = {
    "$defs": {
        "Criterion": {
            "type": "object",
            "properties": {"score": {"type": "integer"}},
            "required": ["score"],
        },
        "Overall": {
            "type": "object",
            "properties": {"score": {"type": "number"}},
            "required": ["score"],
        },
        "Evaluation": {
            "type": "object",
            "properties": {
                "criterion": {"$ref": "#/$defs/Criterion"},
                "overall": {"$ref": "#/$defs/Overall"},
            },
            "required": ["criterion", "overall"],
        },
    },
    "type": "object",
    "properties": {
        "evaluations": {
            "type": "array",
            "items": {"$ref": "#/$defs/Evaluation"},
        }
    },
    "required": ["evaluations"],
}


def _evaluation_scores(overall_score: int | float) -> dict:
    return {"evaluations": [{"criterion": {"score": 9.0}, "overall": {"score": overall_score}}]}


def test_normalize_nested_json_schema_numeric_types() -> None:
    result = validate(_evaluation_scores(9), NESTED_EVALUATION_SCHEMA)

    criterion_score = result["evaluations"][0]["criterion"]["score"]
    overall_score = result["evaluations"][0]["overall"]["score"]
    assert criterion_score == 9
    assert isinstance(criterion_score, int)
    assert overall_score == 9.0
    assert isinstance(overall_score, float)


def test_normalized_nested_numbers_have_compatible_parquet_schemas(tmp_path) -> None:
    for batch_number, score in enumerate((9, 9.5)):
        normalized = validate(_evaluation_scores(score), NESTED_EVALUATION_SCHEMA)
        dataframe = lazy.pd.DataFrame({"qa_evaluations": [normalized]})
        dataframe.to_parquet(tmp_path / f"batch_{batch_number:05d}.parquet", index=False)

    combined = lazy.pd.read_parquet(tmp_path, dtype_backend="pyarrow")
    assert len(combined) == 2
