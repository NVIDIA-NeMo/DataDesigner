# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from copy import deepcopy

import pytest

from data_designer.engine.processing.gsonschema.schema_transformers import forbid_additional_properties


@pytest.fixture
def stub_simple_object_schema():
    return {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }


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
            },
        },
    }


@pytest.fixture
def stub_array_schema():
    return {
        "type": "object",
        "properties": {
            "users": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "name": {"type": "string"},
                    },
                },
            }
        },
    }


def navigate_to_path(result, path):
    """Helper function to navigate to a nested path in the result."""
    if path == "":
        return result
    current = result
    for key in path.split("."):
        if key.isdigit():
            current = current[int(key)]
        else:
            current = current[key]
    return current


@pytest.mark.parametrize(
    "test_case,schema_fixture,expected_additional_properties_paths",
    [
        ("simple_object", "stub_simple_object_schema", [""]),
        ("nested_object", "stub_nested_object_schema", ["", "properties.address"]),
        ("array_with_object_items", "stub_array_schema", ["", "properties.users.items"]),
    ],
)
def test_basic_schema_transformations(request, test_case, schema_fixture, expected_additional_properties_paths):
    schema = request.getfixturevalue(schema_fixture)
    result = forbid_additional_properties(schema)

    for path in expected_additional_properties_paths:
        current = navigate_to_path(result, path)
        assert current["additionalProperties"] is False

    assert result != schema  # Ensure the original wasn't modified


@pytest.mark.parametrize(
    "test_case,schema,expected_additional_properties_paths",
    [
        (
            "array_with_multiple_item_types",
            {
                "type": "object",
                "properties": {
                    "mixedArray": {
                        "type": "array",
                        "items": [
                            {"type": "object", "properties": {"id": {"type": "integer"}}},
                            {"type": "object", "properties": {"name": {"type": "string"}}},
                        ],
                    }
                },
            },
            ["", "properties.mixedArray.items.0", "properties.mixedArray.items.1"],
        ),
        (
            "pattern_properties",
            {
                "type": "object",
                "patternProperties": {
                    "^S_": {"type": "object", "properties": {"value": {"type": "string"}}},
                    "^I_": {"type": "object", "properties": {"value": {"type": "integer"}}},
                },
            },
            ["", "patternProperties.^S_", "patternProperties.^I_"],
        ),
        (
            "allof_anyof_oneof_schemas",
            {
                "type": "object",
                "allOf": [
                    {"type": "object", "properties": {"name": {"type": "string"}}},
                    {"type": "object", "properties": {"email": {"type": "string"}}},
                ],
                "anyOf": [
                    {"type": "object", "properties": {"phone": {"type": "string"}}},
                    {"type": "object", "properties": {"address": {"type": "object"}}},
                ],
                "oneOf": [
                    {"type": "object", "properties": {"type": {"type": "string", "enum": ["personal"]}}},
                    {"type": "object", "properties": {"type": {"type": "string", "enum": ["business"]}}},
                ],
            },
            ["", "allOf.0", "allOf.1", "anyOf.0", "anyOf.1", "oneOf.0", "oneOf.1"],
        ),
        (
            "conditional_schemas",
            {
                "type": "object",
                "properties": {"type": {"type": "string"}},
                "if": {"properties": {"type": {"enum": ["personal"]}}},
                "then": {"properties": {"home_address": {"type": "object"}}},
                "else": {"properties": {"business_address": {"type": "object"}}},
            },
            ["", "then", "else"],
        ),
        (
            "deeply_nested_schema",
            {
                "type": "object",
                "properties": {
                    "level1": {
                        "type": "object",
                        "properties": {
                            "level2": {
                                "type": "object",
                                "properties": {
                                    "level3": {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    }
                                },
                            }
                        },
                    }
                },
            },
            [
                "",
                "properties.level1",
                "properties.level1.properties.level2",
                "properties.level1.properties.level2.properties.level3",
            ],
        ),
    ],
)
def test_complex_schema_transformations(test_case, schema, expected_additional_properties_paths):
    result = forbid_additional_properties(schema)

    for path in expected_additional_properties_paths:
        current = navigate_to_path(result, path)
        assert current["additionalProperties"] is False


@pytest.mark.parametrize(
    "test_case,schema,expected_behavior",
    [
        (
            "schema_with_non_object_types",
            {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
            },
            "non_object_types_unchanged",
        ),
        (
            "nested_not_schema",
            {
                "type": "object",
                "properties": {
                    "data": {
                        "not": {
                            "type": "object",
                            "properties": {"restricted": {"type": "boolean"}},
                        }
                    }
                },
            },
            "not_schema_transformed",
        ),
        (
            "already_has_additional_properties",
            {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "additionalProperties": True,
            },
            "overwrite_existing",
        ),
        (
            "schema_with_properties_but_no_type",
            {"properties": {"name": {"type": "string"}, "age": {"type": "integer"}}},
            "add_additional_properties",
        ),
        (
            "empty_schema",
            {},
            "no_changes",
        ),
    ],
)
def test_special_schema_cases(test_case, schema, expected_behavior):
    result = forbid_additional_properties(schema)

    if expected_behavior == "non_object_types_unchanged":
        assert result["additionalProperties"] is False
        assert "additionalProperties" not in result["properties"]["name"]
        assert "additionalProperties" not in result["properties"]["age"]
        assert "additionalProperties" not in result["properties"]["tags"]
        assert "additionalProperties" not in result["properties"]["tags"]["items"]
    elif expected_behavior == "not_schema_transformed":
        assert result["additionalProperties"] is False
        assert result["properties"]["data"]["not"]["additionalProperties"] is False
    elif expected_behavior == "overwrite_existing":
        assert result["additionalProperties"] is False  # Should be overwritten
    elif expected_behavior == "add_additional_properties":
        assert result["additionalProperties"] is False
    elif expected_behavior == "no_changes":
        assert result == {}  # No changes expected


@pytest.mark.parametrize(
    "test_case,schema,expected_additional_properties_paths",
    [
        (
            "local_ref",
            {
                "type": "object",
                "properties": {"user": {"$ref": "#/$defs/user"}},
                "$defs": {
                    "user": {
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "address": {"type": "object"},
                        },
                    }
                },
            },
            ["", "$defs.user"],
        ),
        (
            "nested_refs",
            {
                "type": "object",
                "properties": {"employee": {"$ref": "#/$defs/employee"}},
                "$defs": {
                    "employee": {
                        "type": "object",
                        "properties": {
                            "details": {"$ref": "#/$defs/person"},
                            "role": {"type": "string"},
                        },
                    },
                    "person": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    },
                },
            },
            ["", "$defs.employee", "$defs.person"],
        ),
        (
            "circular_refs",
            {
                "type": "object",
                "properties": {"node": {"$ref": "#/$defs/node"}},
                "$defs": {
                    "node": {
                        "type": "object",
                        "properties": {
                            "value": {"type": "string"},
                            "child": {"$ref": "#/$defs/node"},  # Reference to itself
                        },
                    }
                },
            },
            ["", "$defs.node"],
        ),
        (
            "array_with_ref_items",
            {
                "type": "object",
                "properties": {"people": {"type": "array", "items": {"$ref": "#/$defs/person"}}},
                "$defs": {
                    "person": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
                    }
                },
            },
            ["", "$defs.person"],
        ),
        (
            "ref_in_combiners",
            {
                "type": "object",
                "properties": {
                    "entity": {
                        "allOf": [
                            {"$ref": "#/$defs/base"},
                            {"type": "object", "properties": {"extra": {"type": "string"}}},
                        ]
                    }
                },
                "$defs": {
                    "base": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string"},
                            "created": {"type": "string", "format": "date-time"},
                        },
                    }
                },
            },
            ["", "properties.entity.allOf.1", "$defs.base"],
        ),
    ],
)
def test_ref_schema_transformations(test_case, schema, expected_additional_properties_paths):
    result = forbid_additional_properties(schema)

    for path in expected_additional_properties_paths:
        current = navigate_to_path(result, path)
        assert current["additionalProperties"] is False

    if test_case == "local_ref":
        assert "$ref" in result["properties"]["user"]  # Ref should be preserved


def test_schema_is_not_modified(stub_simple_object_schema):
    original = deepcopy(stub_simple_object_schema)
    forbid_additional_properties(stub_simple_object_schema)
    assert stub_simple_object_schema == original  # Original should be unchanged
