# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.formatters import (
    format_method_info_json,
    format_method_info_text,
    format_type_list_text,
)
from data_designer.cli.services.introspection.method_inspector import MethodInfo, ParamInfo

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_method(
    name: str = "do_thing",
    signature: str = "do_thing(x: int) -> str",
    description: str = "Does a thing.",
    return_type: str = "str",
    parameters: list[ParamInfo] | None = None,
) -> MethodInfo:
    return MethodInfo(
        name=name,
        signature=signature,
        description=description,
        return_type=return_type,
        parameters=parameters or [ParamInfo(name="x", type_str="int", default=None, description="An integer")],
    )


# ---------------------------------------------------------------------------
# format_method_info_text
# ---------------------------------------------------------------------------


def test_format_method_info_text_basic() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods)
    assert "do_thing(x: int) -> str" in text
    assert "Does a thing." in text
    assert "Parameters:" in text


def test_format_method_info_text_with_class_name() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods, class_name="MyBuilder")
    assert "MyBuilder Methods:" in text


def test_format_method_info_text_no_class_name() -> None:
    methods = [_make_method()]
    text = format_method_info_text(methods, class_name=None)
    assert "Methods:" not in text


# ---------------------------------------------------------------------------
# format_method_info_json
# ---------------------------------------------------------------------------


def test_format_method_info_json_basic() -> None:
    methods = [_make_method()]
    result = format_method_info_json(methods)
    assert isinstance(result, list)
    assert len(result) == 1
    entry = result[0]
    assert entry["name"] == "do_thing"
    assert entry["signature"] == "do_thing(x: int) -> str"
    assert entry["return_type"] == "str"
    assert "description" in entry
    assert "parameters" in entry


def test_format_method_info_json_multiple_methods() -> None:
    methods = [
        _make_method(name="method_a", signature="method_a() -> None", return_type="None", parameters=[]),
        _make_method(name="method_b"),
    ]
    result = format_method_info_json(methods)
    assert len(result) == 2
    names = [e["name"] for e in result]
    assert "method_a" in names
    assert "method_b" in names


# ---------------------------------------------------------------------------
# format_type_list_text
# ---------------------------------------------------------------------------


def test_format_type_list_text_basic() -> None:
    class FakeA:
        pass

    class FakeB:
        pass

    items: dict[str, type] = {"alpha": FakeA, "beta": FakeB}
    text = format_type_list_text(items, "type_name", "class_name")
    assert "type_name" in text
    assert "class_name" in text
    assert "alpha" in text
    assert "FakeA" in text
    assert "beta" in text
    assert "FakeB" in text


def test_format_type_list_text_alignment() -> None:
    class C:
        pass

    items: dict[str, type] = {"short": C, "very_long_name": C}
    text = format_type_list_text(items, "Type", "Class")
    lines = text.strip().split("\n")
    # Header + separator + 2 data rows
    assert len(lines) == 4


def test_format_type_list_text_empty() -> None:
    text = format_type_list_text({}, "Type", "Class")
    assert "(no items)" in text
