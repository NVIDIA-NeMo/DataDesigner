# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.services.introspection.method_inspector import (
    MethodInfo,
    _is_dunder,
    _is_private,
    _parse_google_docstring_args,
    inspect_class_methods,
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


# ---------------------------------------------------------------------------
# inspect_class_methods — edge cases (P1-4)
# ---------------------------------------------------------------------------


class EmptyClass:
    """A class with no public methods (no __init__ either)."""



class ClassWithBadSignature:
    """A class where one method has an uninspectable signature."""

    def good_method(self) -> str:
        """Works fine."""
        return "ok"


class ClassWithVarArgs:
    """A class with *args and **kwargs."""

    def method_with_varargs(self, *args: str, **kwargs: int) -> None:
        """A method with varargs."""


def test_inspect_class_methods_empty_class() -> None:
    methods = inspect_class_methods(EmptyClass, include_private=False)
    assert methods == []


def test_inspect_class_methods_signature_error_skipped() -> None:
    import inspect as _inspect
    from unittest.mock import patch

    original_sig = _inspect.signature

    def bad_signature(method: object) -> _inspect.Signature:
        if getattr(method, "__name__", "") == "good_method":
            raise ValueError("cannot inspect")
        return original_sig(method)

    with patch(
        "data_designer.cli.services.introspection.method_inspector.inspect.signature", side_effect=bad_signature
    ):
        methods = inspect_class_methods(ClassWithBadSignature, include_private=False)

    names = [m.name for m in methods]
    assert "good_method" not in names


def test_inspect_class_methods_varargs_and_kwargs() -> None:
    methods = inspect_class_methods(ClassWithVarArgs, include_private=False)
    m = next(m for m in methods if m.name == "method_with_varargs")
    assert "*args" in m.signature
    assert "**kwargs" in m.signature


# ---------------------------------------------------------------------------
# _is_dunder / _is_private (P2-1)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("__init__", False),
        ("__str__", True),
        ("__repr__", True),
        ("regular", False),
        ("_private", False),
    ],
)
def test_is_dunder(name: str, expected: bool) -> None:
    assert _is_dunder(name) is expected


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("_foo", True),
        ("_private_method", True),
        ("__init__", False),
        ("__str__", False),
        ("public", False),
    ],
)
def test_is_private(name: str, expected: bool) -> None:
    assert _is_private(name) is expected


# ---------------------------------------------------------------------------
# keyword-only params (P2-9)
# ---------------------------------------------------------------------------


class ClassWithKeywordOnly:
    """A class with keyword-only parameters."""

    def method_with_kw(self, *, kw: str = "x") -> None:
        """A method with keyword-only arg."""


def test_format_signature_keyword_only() -> None:
    methods = inspect_class_methods(ClassWithKeywordOnly, include_private=False)
    m = next(m for m in methods if m.name == "method_with_kw")
    assert "*, " in m.signature or "*," in m.signature
