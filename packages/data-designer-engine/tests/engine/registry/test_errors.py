# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from data_designer.engine.errors import DataDesignerError
from data_designer.engine.registry.errors import (
    NotFoundInRegistryError,
    RegistryItemNotTypeError,
)


def test_not_found_in_registry_error_creation():
    error = NotFoundInRegistryError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, DataDesignerError)


def test_not_found_in_registry_error_with_custom_message():
    message = "Custom not found error"
    error = NotFoundInRegistryError(message)
    assert str(error) == message


def test_not_found_in_registry_error_attributes():
    error = NotFoundInRegistryError("Test")
    assert hasattr(error, "__dict__")
    assert hasattr(error, "__str__")


def test_registry_item_not_type_error_creation():
    error = RegistryItemNotTypeError("Test error message")
    assert str(error) == "Test error message"
    assert isinstance(error, DataDesignerError)


def test_registry_item_not_type_error_with_custom_message():
    message = "Custom type error"
    error = RegistryItemNotTypeError(message)
    assert str(error) == message


def test_registry_item_not_type_error_attributes():
    error = RegistryItemNotTypeError("Test")
    assert hasattr(error, "__dict__")
    assert hasattr(error, "__str__")


def test_registry_item_not_type_error_raising():
    with pytest.raises(RegistryItemNotTypeError, match="Test error"):
        raise RegistryItemNotTypeError("Test error")


def test_error_hierarchy_error_types_are_different():
    assert NotFoundInRegistryError is not RegistryItemNotTypeError


def test_error_hierarchy_error_instances_are_different_types():
    not_found = NotFoundInRegistryError("test")
    not_type = RegistryItemNotTypeError("test")

    assert not isinstance(not_found, RegistryItemNotTypeError)
    assert not isinstance(not_type, NotFoundInRegistryError)
