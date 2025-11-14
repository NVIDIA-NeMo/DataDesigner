# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.cli.forms.field import Field, NumericField, SelectField, TextField, ValidationError


# ValidationError tests
def test_validation_error_is_exception() -> None:
    """Test that ValidationError is an exception."""
    error = ValidationError("Test error")
    assert isinstance(error, Exception)
    assert str(error) == "Test error"


# TextField tests - focus on validation behavior
def test_text_field_value_setter_without_validator() -> None:
    """Test setting TextField value without validator succeeds."""
    field = TextField(name="name", prompt="Enter name")
    field.value = "John Doe"
    assert field.value == "John Doe"


def test_text_field_value_setter_with_valid_value() -> None:
    """Test setting TextField value with validator that passes."""
    validator = Mock(return_value=(True, None))
    field = TextField(name="email", prompt="Enter email", validator=validator)

    field.value = "test@example.com"

    assert field.value == "test@example.com"
    validator.assert_called_once_with("test@example.com")


def test_text_field_value_setter_with_invalid_value() -> None:
    """Test setting TextField value with validator that fails raises ValidationError."""
    validator = Mock(return_value=(False, "Invalid format"))
    field = TextField(name="email", prompt="Enter email", validator=validator)

    with pytest.raises(ValidationError, match="Invalid format"):
        field.value = "not-an-email"

    assert field.value is None


def test_text_field_validator_receives_string() -> None:
    """Test that validator always receives string values."""
    validator = Mock(return_value=(True, None))
    field = TextField(name="field", prompt="Enter", validator=validator)

    field.value = "text"
    validator.assert_called_with("text")


@patch("data_designer.cli.ui.prompt_text_input")
def test_text_field_prompt_user_returns_input(mock_prompt: Mock) -> None:
    """Test TextField prompt_user returns user input."""
    mock_prompt.return_value = "user input"
    field = TextField(name="name", prompt="Enter name")

    assert field.prompt_user() == "user input"


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
@patch("data_designer.cli.ui.prompt_text_input")
def test_text_field_prompt_user_handles_back_navigation(mock_prompt: Mock) -> None:
    """Test TextField prompt_user properly returns BACK sentinel."""
    mock_prompt.return_value = "BACK_SENTINEL"
    field = TextField(name="name", prompt="Enter name")

    result = field.prompt_user(allow_back=True)

    assert result == "BACK_SENTINEL"


# SelectField tests
def test_select_field_value_setter() -> None:
    """Test setting SelectField value."""
    options = {"1": "One", "2": "Two"}
    field = SelectField(name="number", prompt="Select number", options=options)

    field.value = "1"

    assert field.value == "1"


@patch("data_designer.cli.ui.select_with_arrows")
def test_select_field_prompt_user_returns_selection(mock_select: Mock) -> None:
    """Test SelectField prompt_user returns user selection."""
    mock_select.return_value = "opt1"
    options = {"opt1": "Option 1", "opt2": "Option 2"}
    field = SelectField(name="choice", prompt="Select", options=options)

    assert field.prompt_user() == "opt1"


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
@patch("data_designer.cli.ui.select_with_arrows")
def test_select_field_prompt_user_handles_back_navigation(mock_select: Mock) -> None:
    """Test SelectField prompt_user properly returns BACK sentinel."""
    mock_select.return_value = "BACK_SENTINEL"
    options = {"1": "One", "2": "Two"}
    field = SelectField(name="num", prompt="Select", options=options)

    result = field.prompt_user(allow_back=True)

    assert result == "BACK_SENTINEL"


# NumericField validator tests - core business logic
def test_numeric_field_validator_valid_value() -> None:
    """Test NumericField validator accepts valid value within range."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    is_valid, error = field.validator("25")

    assert is_valid is True
    assert error is None


def test_numeric_field_validator_rejects_below_min() -> None:
    """Test NumericField validator rejects value below minimum."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    is_valid, error = field.validator("-5")

    assert is_valid is False
    assert error == "Value must be between 0.0 and 150.0"


def test_numeric_field_validator_rejects_above_max() -> None:
    """Test NumericField validator rejects value above maximum."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    is_valid, error = field.validator("200")

    assert is_valid is False
    assert error == "Value must be between 0.0 and 150.0"


def test_numeric_field_validator_only_min_accepts_valid() -> None:
    """Test NumericField validator with only min value accepts valid input."""
    field = NumericField(name="count", prompt="Enter count", min_value=10.0)

    is_valid, error = field.validator("15")

    assert is_valid is True
    assert error is None


def test_numeric_field_validator_only_min_rejects_invalid() -> None:
    """Test NumericField validator with only min value rejects invalid input."""
    field = NumericField(name="count", prompt="Enter count", min_value=10.0)

    is_valid, error = field.validator("5")

    assert is_valid is False
    assert error == "Value must be >= 10.0"


def test_numeric_field_validator_only_max_accepts_valid() -> None:
    """Test NumericField validator with only max value accepts valid input."""
    field = NumericField(name="score", prompt="Enter score", max_value=100.0)

    is_valid, error = field.validator("85")

    assert is_valid is True
    assert error is None


def test_numeric_field_validator_only_max_rejects_invalid() -> None:
    """Test NumericField validator with only max value rejects invalid input."""
    field = NumericField(name="score", prompt="Enter score", max_value=100.0)

    is_valid, error = field.validator("150")

    assert is_valid is False
    assert error == "Value must be <= 100.0"


def test_numeric_field_validator_rejects_non_numeric_with_range() -> None:
    """Test NumericField validator rejects non-numeric input when range is set."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    is_valid, error = field.validator("not-a-number")

    assert is_valid is False
    assert error == "Value must be between 0.0 and 150.0"


def test_numeric_field_validator_rejects_non_numeric_without_range() -> None:
    """Test NumericField validator rejects non-numeric input when no range is set."""
    field = NumericField(name="count", prompt="Enter count")

    is_valid, error = field.validator("not-a-number")

    assert is_valid is False
    assert error == "Must be a valid number"


def test_numeric_field_validator_accepts_empty_when_not_required() -> None:
    """Test NumericField validator accepts empty value when field is not required."""
    field = NumericField(
        name="optional_value",
        prompt="Enter value",
        required=False,
        min_value=0.0,
        max_value=100.0,
    )

    is_valid, error = field.validator("")

    assert is_valid is True
    assert error is None


def test_numeric_field_validator_handles_boundary_values() -> None:
    """Test NumericField validator accepts boundary values."""
    field = NumericField(name="score", prompt="Enter score", min_value=0.0, max_value=100.0)

    # Test min boundary
    is_valid_min, error_min = field.validator("0.0")
    assert is_valid_min is True
    assert error_min is None

    # Test max boundary
    is_valid_max, error_max = field.validator("100.0")
    assert is_valid_max is True
    assert error_max is None


# NumericField value setter tests with validation
def test_numeric_field_value_setter_accepts_valid() -> None:
    """Test setting NumericField value with valid number succeeds."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    field.value = 25.5

    assert field.value == 25.5


def test_numeric_field_value_setter_rejects_invalid() -> None:
    """Test setting NumericField value with invalid number raises ValidationError."""
    field = NumericField(name="age", prompt="Enter age", min_value=0.0, max_value=150.0)

    with pytest.raises(ValidationError):
        field.value = 200.0

    assert field.value is None


# NumericField prompt_user tests
@patch("data_designer.cli.ui.prompt_text_input")
def test_numeric_field_prompt_user_returns_float(mock_prompt: Mock) -> None:
    """Test NumericField prompt_user converts string to float."""
    mock_prompt.return_value = "42"
    field = NumericField(name="age", prompt="Enter age")

    result = field.prompt_user()

    assert result == 42.0
    assert isinstance(result, float)


@patch("data_designer.cli.ui.prompt_text_input")
def test_numeric_field_prompt_user_returns_none_for_empty(mock_prompt: Mock) -> None:
    """Test NumericField prompt_user returns None for empty input."""
    mock_prompt.return_value = ""
    field = NumericField(name="optional", prompt="Enter value", required=False)

    result = field.prompt_user()

    assert result is None


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
@patch("data_designer.cli.ui.prompt_text_input")
def test_numeric_field_prompt_user_handles_back_navigation(mock_prompt: Mock) -> None:
    """Test NumericField prompt_user properly returns BACK sentinel."""
    mock_prompt.return_value = "BACK_SENTINEL"
    field = NumericField(name="value", prompt="Enter value")

    result = field.prompt_user(allow_back=True)

    assert result == "BACK_SENTINEL"


# Field base class tests - design constraints
def test_field_is_abstract() -> None:
    """Test that Field cannot be instantiated directly."""
    with pytest.raises(TypeError):
        Field(name="test", prompt="Test prompt")  # type: ignore


def test_field_generic_type_preserved() -> None:
    """Test that Field generic type is preserved in subclasses."""
    text_field = TextField(name="text", prompt="Enter text")
    numeric_field = NumericField(name="num", prompt="Enter number")

    text_field.value = "string value"
    numeric_field.value = 42.0

    assert isinstance(text_field.value, str)
    assert isinstance(numeric_field.value, float)


def test_validator_converts_non_string_values() -> None:
    """Test that validator converts non-string values to strings before validation."""
    validator = Mock(return_value=(True, None))
    field = NumericField(name="num", prompt="Enter number", min_value=0.0, max_value=100.0)

    # Override validator to test conversion
    field.validator = validator
    field.value = 42.5

    # Validator should be called with string representation
    validator.assert_called_once_with("42.5")
