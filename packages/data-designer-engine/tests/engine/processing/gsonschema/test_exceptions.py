# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from jsonschema import ValidationError

from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError


@pytest.mark.parametrize(
    "test_case,message,expected_message",
    [
        ("basic_error", "Test error message", "Test error message"),
        ("custom_message", "Custom validation error", "Custom validation error"),
        ("empty_message", "", ""),
    ],
)
def test_json_schema_validation_error_creation(test_case, message, expected_message):
    error = JSONSchemaValidationError(message)
    assert str(error) == expected_message
    assert isinstance(error, ValidationError)
    assert isinstance(error, JSONSchemaValidationError)


def test_json_schema_validation_error_attributes():
    error = JSONSchemaValidationError("Test")
    assert hasattr(error, "__dict__")
    assert hasattr(error, "__str__")


def test_json_schema_validation_error_raising():
    with pytest.raises(JSONSchemaValidationError, match="Test error"):
        raise JSONSchemaValidationError("Test error")


def test_json_schema_validation_error_alias_compatibility():
    error1 = ValidationError("test")
    error2 = JSONSchemaValidationError("test")

    assert isinstance(error1, ValidationError)
    assert isinstance(error2, ValidationError)
    assert isinstance(error2, JSONSchemaValidationError)
