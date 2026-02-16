# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.method_inspector import (
    MethodInfo,
    PropertyInfo,
    _parse_google_docstring_args,
    inspect_class_methods,
    inspect_class_properties,
)

# ---------------------------------------------------------------------------
# Test helper classes
# ---------------------------------------------------------------------------


class SampleClass:
    """A sample class for testing method introspection."""

    def public_method(self, x: int, y: str = "default") -> str:
        """Do something public.

        Args:
            x: The integer input.
            y: An optional string.

        Returns:
            A result string.
        """
        return f"{x}-{y}"

    def another_public(self) -> None:
        """Another public method with no args."""

    def _private_method(self, z: float) -> float:
        """A private helper.

        Args:
            z: A float value.
        """
        return z * 2

    def __dunder_method__(self) -> None:
        """Should be excluded (dunder)."""

    def __init__(self) -> None:
        """Init should be included."""


class ClassWithClassmethod:
    """A class with a classmethod for testing."""

    @classmethod
    def from_value(cls, value: int) -> ClassWithClassmethod:
        """Create an instance from a value.

        Args:
            value: The input value.
        """
        return cls()

    def regular_method(self) -> str:
        """A regular method."""
        return "hello"


class ClassWithProperties:
    """A class with properties for testing."""

    def __init__(self) -> None:
        self._name = "test"
        self._count = 0

    @property
    def name(self) -> str:
        """Get the name."""
        return self._name

    @property
    def count(self) -> int:
        """Get the count value."""
        return self._count

    @property
    def _private_prop(self) -> bool:
        """A private property."""
        return True


class ClassWithDefaultInitDocstring:
    """A useful class that does important things.

    This is a longer description of the class.
    """

    def __init__(self, x: int = 0) -> None:
        self.x = x


class ClassWithCustomInitDocstring:
    """Class-level docstring."""

    def __init__(self, x: int) -> None:
        """Custom init docstring.

        Args:
            x: An integer.
        """
        self.x = x


# ---------------------------------------------------------------------------
# _parse_google_docstring_args
# ---------------------------------------------------------------------------


def test_parse_google_docstring_args_basic() -> None:
    docstring = """Do something.

    Args:
        x: The first parameter.
        y: The second parameter.

    Returns:
        A result.
    """
    result = _parse_google_docstring_args(docstring)
    assert "x" in result
    assert result["x"] == "The first parameter."
    assert "y" in result
    assert result["y"] == "The second parameter."


def test_parse_google_docstring_args_empty() -> None:
    assert _parse_google_docstring_args(None) == {}
    assert _parse_google_docstring_args("") == {}


def test_parse_google_docstring_args_no_args_section() -> None:
    docstring = """Just a description.

    Returns:
        Something.
    """
    result = _parse_google_docstring_args(docstring)
    assert result == {}


def test_parse_google_docstring_args_multiline_description() -> None:
    docstring = """Do something.

    Args:
        x: First line of description
            continued on second line.
        y: Another param.
    """
    result = _parse_google_docstring_args(docstring)
    assert "x" in result
    assert "continued" in result["x"]
    assert "y" in result


# ---------------------------------------------------------------------------
# inspect_class_methods - exclude private
# ---------------------------------------------------------------------------


def test_inspect_class_methods_public_only() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    names = [m.name for m in methods]
    assert "public_method" in names
    assert "another_public" in names
    assert "_private_method" not in names
    assert "__dunder_method__" not in names


def test_inspect_class_methods_returns_method_info() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    assert all(isinstance(m, MethodInfo) for m in methods)


def test_inspect_class_methods_signature_content() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    public = next(m for m in methods if m.name == "public_method")
    assert "x: int" in public.signature
    assert "y: str" in public.signature
    assert "str" in public.return_type


def test_inspect_class_methods_description() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    public = next(m for m in methods if m.name == "public_method")
    assert public.description == "Do something public."


def test_inspect_class_methods_parameters() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    public = next(m for m in methods if m.name == "public_method")
    param_names = [p.name for p in public.parameters]
    assert "x" in param_names
    assert "y" in param_names
    x_param = next(p for p in public.parameters if p.name == "x")
    assert x_param.description == "The integer input."


# ---------------------------------------------------------------------------
# inspect_class_methods - include private
# ---------------------------------------------------------------------------


def test_inspect_class_methods_include_private() -> None:
    methods = inspect_class_methods(SampleClass, include_private=True)
    names = [m.name for m in methods]
    assert "_private_method" in names
    assert "__dunder_method__" not in names


def test_inspect_class_methods_init_included() -> None:
    methods = inspect_class_methods(SampleClass, include_private=True)
    names = [m.name for m in methods]
    assert "__init__" in names


# ---------------------------------------------------------------------------
# inspect_class_methods - classmethod detection
# ---------------------------------------------------------------------------


def test_inspect_class_methods_detects_classmethod() -> None:
    methods = inspect_class_methods(ClassWithClassmethod, include_private=False)
    names = [m.name for m in methods]
    assert "from_value" in names
    assert "regular_method" in names


def test_inspect_class_methods_classmethod_is_classmethod_flag() -> None:
    methods = inspect_class_methods(ClassWithClassmethod, include_private=False)
    from_value = next(m for m in methods if m.name == "from_value")
    regular = next(m for m in methods if m.name == "regular_method")
    assert from_value.is_classmethod is True
    assert regular.is_classmethod is False


def test_inspect_class_methods_classmethod_signature() -> None:
    methods = inspect_class_methods(ClassWithClassmethod, include_private=False)
    from_value = next(m for m in methods if m.name == "from_value")
    assert "value: int" in from_value.signature


def test_inspect_class_methods_classmethod_description() -> None:
    methods = inspect_class_methods(ClassWithClassmethod, include_private=False)
    from_value = next(m for m in methods if m.name == "from_value")
    assert from_value.description == "Create an instance from a value."


def test_inspect_class_methods_classmethod_parameters() -> None:
    methods = inspect_class_methods(ClassWithClassmethod, include_private=False)
    from_value = next(m for m in methods if m.name == "from_value")
    param_names = [p.name for p in from_value.parameters]
    assert "value" in param_names
    value_param = next(p for p in from_value.parameters if p.name == "value")
    assert value_param.description == "The input value."


def test_inspect_class_methods_regular_not_classmethod() -> None:
    methods = inspect_class_methods(SampleClass, include_private=False)
    for m in methods:
        assert m.is_classmethod is False


# ---------------------------------------------------------------------------
# inspect_class_properties
# ---------------------------------------------------------------------------


def test_inspect_class_properties_finds_public() -> None:
    props = inspect_class_properties(ClassWithProperties, include_private=False)
    names = [p.name for p in props]
    assert "name" in names
    assert "count" in names
    assert "_private_prop" not in names


def test_inspect_class_properties_returns_property_info() -> None:
    props = inspect_class_properties(ClassWithProperties, include_private=False)
    assert all(isinstance(p, PropertyInfo) for p in props)


def test_inspect_class_properties_return_types() -> None:
    props = inspect_class_properties(ClassWithProperties, include_private=False)
    name_prop = next(p for p in props if p.name == "name")
    count_prop = next(p for p in props if p.name == "count")
    assert name_prop.return_type == "str"
    assert count_prop.return_type == "int"


def test_inspect_class_properties_descriptions() -> None:
    props = inspect_class_properties(ClassWithProperties, include_private=False)
    name_prop = next(p for p in props if p.name == "name")
    count_prop = next(p for p in props if p.name == "count")
    assert name_prop.description == "Get the name."
    assert count_prop.description == "Get the count value."


def test_inspect_class_properties_include_private() -> None:
    props = inspect_class_properties(ClassWithProperties, include_private=True)
    names = [p.name for p in props]
    assert "_private_prop" in names


def test_inspect_class_properties_empty_class() -> None:
    props = inspect_class_properties(SampleClass, include_private=False)
    assert props == []


# ---------------------------------------------------------------------------
# __init__ docstring fallback
# ---------------------------------------------------------------------------


def test_init_default_docstring_falls_back_to_class() -> None:
    methods = inspect_class_methods(ClassWithDefaultInitDocstring, include_private=True)
    init = next((m for m in methods if m.name == "__init__"), None)
    assert init is not None
    assert init.description == "A useful class that does important things."


def test_init_custom_docstring_preserved() -> None:
    methods = inspect_class_methods(ClassWithCustomInitDocstring, include_private=True)
    init = next((m for m in methods if m.name == "__init__"), None)
    assert init is not None
    assert init.description == "Custom init docstring."
