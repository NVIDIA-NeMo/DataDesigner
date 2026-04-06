# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from jinja2 import StrictUndefined

from data_designer.engine.dataset_builders.utils.skip_evaluator import (
    NativeSandboxedEnvironment,
    evaluate_skip_when,
    should_skip_by_propagation,
)


def test_native_sandboxed_environment_returns_native_types() -> None:
    env = NativeSandboxedEnvironment(undefined=StrictUndefined)
    result = env.from_string("{{ 1 + 1 }}").render()
    assert result == 2
    assert type(result) is int


def test_evaluate_skip_when_truthy() -> None:
    assert evaluate_skip_when("{{ x == 0 }}", {"x": 0}) is True


def test_evaluate_skip_when_falsy() -> None:
    assert evaluate_skip_when("{{ x == 0 }}", {"x": 1}) is False


def test_evaluate_skip_when_native_false() -> None:
    assert evaluate_skip_when("{{ x }}", {"x": False}) is False


def test_evaluate_skip_when_native_none() -> None:
    assert evaluate_skip_when("{{ x }}", {"x": None}) is False


def test_evaluate_skip_when_native_zero() -> None:
    assert evaluate_skip_when("{{ x }}", {"x": 0}) is False


def test_evaluate_skip_when_native_empty_string() -> None:
    assert evaluate_skip_when("{{ x }}", {"x": ""}) is False


def test_evaluate_skip_when_deserializes_json() -> None:
    assert evaluate_skip_when('{{ x.key == "val" }}', {"x": '{"key": "val"}'}) is True


def test_evaluate_skip_when_strict_undefined_returns_true() -> None:
    """Missing variables trigger fail-safe: returns True (skip the row) and logs a warning."""
    assert evaluate_skip_when("{{ missing_var }}", {}) is True


def test_should_skip_by_propagation_true() -> None:
    assert should_skip_by_propagation(["a", "b"], {"a"}) is True


def test_should_skip_by_propagation_no_overlap() -> None:
    assert should_skip_by_propagation(["a"], {"b"}) is False


def test_should_skip_by_propagation_empty_required() -> None:
    assert should_skip_by_propagation([], {"a"}) is False


def test_should_skip_by_propagation_empty_skipped() -> None:
    assert should_skip_by_propagation(["a"], set()) is False
