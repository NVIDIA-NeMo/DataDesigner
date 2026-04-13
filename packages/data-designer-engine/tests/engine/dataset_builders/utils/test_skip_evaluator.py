# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from jinja2 import StrictUndefined

from data_designer.engine.dataset_builders.utils.skip_evaluator import (
    NativeSandboxedEnvironment,
    evaluate_skip_when,
    should_skip_by_propagation,
    should_skip_column_for_record,
)
from data_designer.engine.dataset_builders.utils.skip_tracker import SKIPPED_COLUMNS_RECORD_KEY


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


# -- should_skip_column_for_record (unified decision) -----------------------

_UPSTREAM_SKIPPED_RECORD: dict = {SKIPPED_COLUMNS_RECORD_KEY: {"upstream_col"}, "upstream_col": None}


@pytest.mark.parametrize(
    ("record", "propagate_skip", "required_columns", "skip_config_when", "expected"),
    [
        pytest.param(
            {"x": 1},
            True,
            [],
            None,
            False,
            id="no-gate-no-propagation",
        ),
        pytest.param(
            _UPSTREAM_SKIPPED_RECORD,
            True,
            ["upstream_col"],
            None,
            True,
            id="propagation-triggers-on-upstream-skip",
        ),
        pytest.param(
            _UPSTREAM_SKIPPED_RECORD,
            False,
            ["upstream_col"],
            None,
            False,
            id="propagation-disabled-ignores-upstream-skip",
        ),
        pytest.param(
            {"gate": 0},
            True,
            [],
            "{{ gate == 0 }}",
            True,
            id="expression-truthy-skips",
        ),
        pytest.param(
            {"gate": 1},
            True,
            [],
            "{{ gate == 0 }}",
            False,
            id="expression-falsy-does-not-skip",
        ),
        pytest.param(
            {SKIPPED_COLUMNS_RECORD_KEY: {"dep"}, "dep": None, "gate": 999},
            True,
            ["dep"],
            "{{ gate == 0 }}",
            True,
            id="propagation-short-circuits-before-expression",
        ),
        pytest.param(
            {SKIPPED_COLUMNS_RECORD_KEY: {"other"}, "gate": 0},
            True,
            ["dep"],
            "{{ gate == 0 }}",
            True,
            id="expression-evaluated-when-propagation-does-not-trigger",
        ),
        pytest.param(
            {"gate": 1},
            True,
            ["dep"],
            "{{ gate == 0 }}",
            False,
            id="both-propagation-and-expression-false",
        ),
    ],
)
def test_should_skip_column_for_record(
    record: dict,
    propagate_skip: bool,
    required_columns: list[str],
    skip_config_when: str | None,
    expected: bool,
) -> None:
    assert (
        should_skip_column_for_record(
            record,
            propagate_skip=propagate_skip,
            required_columns=required_columns,
            skip_config_when=skip_config_when,
        )
        is expected
    )
