# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.method_inspector import (
    MethodInfo,
    _parse_google_docstring_args,
    inspect_class_methods,
)

# ---------------------------------------------------------------------------
# Test helper class
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
