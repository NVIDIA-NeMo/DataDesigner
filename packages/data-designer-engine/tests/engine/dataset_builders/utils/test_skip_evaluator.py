# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
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


@pytest.mark.parametrize(
    ("expression", "record", "expected"),
    [
        pytest.param("{{ x == 0 }}", {"x": 0}, True, id="truthy-match"),
        pytest.param("{{ x == 0 }}", {"x": 1}, False, id="falsy-no-match"),
        pytest.param("{{ x }}", {"x": False}, False, id="native-false"),
        pytest.param("{{ x }}", {"x": None}, False, id="native-none"),
        pytest.param("{{ x }}", {"x": 0}, False, id="native-zero"),
        pytest.param("{{ x }}", {"x": ""}, False, id="native-empty-string"),
        pytest.param('{{ x.key == "val" }}', {"x": '{"key": "val"}'}, True, id="deserializes-json"),
    ],
)
def test_evaluate_skip_when(expression: str, record: dict, expected: bool) -> None:
    assert evaluate_skip_when(expression, record) is expected


def test_evaluate_skip_when_strict_undefined_returns_true() -> None:
    """Missing variables trigger fail-safe: returns True (skip the row) and logs a warning."""
    assert evaluate_skip_when("{{ missing_var }}", {}) is True


@pytest.mark.parametrize(
    ("required", "skipped", "expected"),
    [
        pytest.param(["a", "b"], {"a"}, True, id="overlap"),
        pytest.param(["a"], {"b"}, False, id="no-overlap"),
        pytest.param([], {"a"}, False, id="empty-required"),
        pytest.param(["a"], set(), False, id="empty-skipped"),
    ],
)
def test_should_skip_by_propagation(required: list[str], skipped: set[str], expected: bool) -> None:
    assert should_skip_by_propagation(required, skipped) is expected
