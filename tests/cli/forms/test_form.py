# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import Mock, patch

import pytest

from data_designer.cli.forms.field import TextField, ValidationError
from data_designer.cli.forms.form import Form


# Helper to create a simple field for testing
def create_field(name: str, prompt: str = "Enter value") -> TextField:
    """Create a TextField for testing."""
    return TextField(name=name, prompt=prompt)


# get_field tests - basic lookup behavior
def test_get_field_returns_existing_field() -> None:
    """Test get_field returns field when it exists."""
    field1 = create_field("name")
    field2 = create_field("email")
    form = Form(name="test_form", fields=[field1, field2])

    assert form.get_field("name") is field1
    assert form.get_field("email") is field2


def test_get_field_returns_none_for_nonexistent_field() -> None:
    """Test get_field returns None when field doesn't exist."""
    field = create_field("name")
    form = Form(name="test_form", fields=[field])

    result = form.get_field("nonexistent")

    assert result is None


def test_field_map_handles_duplicate_names() -> None:
    """Test field map with duplicate names - last one wins."""
    field1 = create_field("name")
    field2 = create_field("name")

    form = Form(name="test_form", fields=[field1, field2])

    # Last field with same name should be accessible
    assert form.get_field("name") is field2


# get_values tests - filtering and collection behavior
def test_get_values_returns_only_non_none_values() -> None:
    """Test get_values filters out fields with None values."""
    field1 = create_field("name")
    field2 = create_field("email")
    field3 = create_field("optional")

    field1.value = "John"
    field2.value = "john@example.com"
    field3.value = None

    form = Form(name="test_form", fields=[field1, field2, field3])

    result = form.get_values()

    assert result == {"name": "John", "email": "john@example.com"}
    assert "optional" not in result


def test_get_values_returns_empty_dict_when_no_values_set() -> None:
    """Test get_values returns empty dict when all fields are None."""
    field1 = create_field("name")
    field2 = create_field("email")
    form = Form(name="test_form", fields=[field1, field2])

    result = form.get_values()

    assert result == {}


# set_values tests - batch setting behavior
def test_set_values_sets_matching_fields() -> None:
    """Test set_values updates all matching fields."""
    field1 = create_field("name")
    field2 = create_field("email")
    form = Form(name="test_form", fields=[field1, field2])

    form.set_values({"name": "Alice", "email": "alice@example.com"})

    assert field1.value == "Alice"
    assert field2.value == "alice@example.com"


def test_set_values_ignores_unknown_fields() -> None:
    """Test set_values silently ignores values for non-existent fields."""
    field = create_field("name")
    form = Form(name="test_form", fields=[field])

    # Should not raise error
    form.set_values({"name": "Bob", "unknown": "value"})

    assert field.value == "Bob"


def test_set_values_triggers_field_validation() -> None:
    """Test set_values enforces field validators."""
    validator = Mock(return_value=(False, "Invalid value"))
    field = TextField(name="email", prompt="Enter email", validator=validator)
    form = Form(name="test_form", fields=[field])

    with pytest.raises(ValidationError):
        form.set_values({"email": "invalid"})


# prompt_all tests - focus on navigation logic and edge cases
@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_collects_all_values_sequentially() -> None:
    """Test prompt_all prompts each field in order and returns all values."""
    field1 = create_field("name")
    field2 = create_field("email")
    field3 = create_field("phone")

    field1.prompt_user = Mock(return_value="John")
    field2.prompt_user = Mock(return_value="john@example.com")
    field3.prompt_user = Mock(return_value="123-456-7890")

    form = Form(name="test_form", fields=[field1, field2, field3])

    result = form.prompt_all()

    assert result == {
        "name": "John",
        "email": "john@example.com",
        "phone": "123-456-7890",
    }


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_returns_none_when_user_cancels() -> None:
    """Test prompt_all returns None when any field returns None (cancel)."""
    field1 = create_field("name")
    field2 = create_field("email")

    field1.prompt_user = Mock(return_value="John")
    field2.prompt_user = Mock(return_value=None)

    form = Form(name="test_form", fields=[field1, field2])

    result = form.prompt_all()

    assert result is None


@patch("data_designer.cli.forms.form.print_error")
@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_retries_on_validation_error(mock_print_error: Mock) -> None:
    """Test prompt_all catches validation errors and re-prompts user."""
    validator = Mock(
        side_effect=[
            (False, "Invalid format"),
            (True, None),
        ]
    )
    field = TextField(name="email", prompt="Enter email", validator=validator)
    field.prompt_user = Mock(side_effect=["invalid", "valid@example.com"])

    form = Form(name="test_form", fields=[field])

    result = form.prompt_all()

    # Should succeed after retry
    assert result == {"email": "valid@example.com"}
    # Should have printed error message
    assert mock_print_error.called
    # Should have prompted twice
    assert field.prompt_user.call_count == 2


# Edge cases
@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_with_single_field() -> None:
    """Test prompt_all works with single field form."""
    field = create_field("name")
    field.prompt_user = Mock(return_value="John")

    form = Form(name="test_form", fields=[field])

    result = form.prompt_all()

    assert result == {"name": "John"}


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_with_empty_fields_list() -> None:
    """Test prompt_all with no fields returns empty dict."""
    form = Form(name="test_form", fields=[])

    result = form.prompt_all()

    assert result == {}


@patch("data_designer.cli.ui.BACK", "BACK_SENTINEL")
def test_prompt_all_preserves_values_from_interrupted_session() -> None:
    """Test that values are stored even if user cancels later."""
    field1 = create_field("name")
    field2 = create_field("email")
    field3 = create_field("phone")

    field1.prompt_user = Mock(return_value="John")
    field2.prompt_user = Mock(return_value="john@example.com")
    field3.prompt_user = Mock(return_value=None)  # User cancels

    form = Form(name="test_form", fields=[field1, field2, field3])

    result = form.prompt_all()

    # Should return None on cancel
    assert result is None
    # But fields should still have their values
    assert field1.value == "John"
    assert field2.value == "john@example.com"
    assert field3.value is None
