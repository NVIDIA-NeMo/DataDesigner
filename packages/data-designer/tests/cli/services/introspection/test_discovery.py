# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.discovery import (
    discover_column_configs,
    discover_constraint_types,
    discover_mcp_types,
    discover_model_configs,
    discover_processor_configs,
    discover_sampler_types,
    discover_seed_types,
    discover_validator_types,
)

# ---------------------------------------------------------------------------
# discover_column_configs
# ---------------------------------------------------------------------------


def test_discover_column_configs_returns_dict() -> None:
    result = discover_column_configs()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_column_configs_contains_expected_keys() -> None:
    result = discover_column_configs()
    for expected_key in ("llm-text", "sampler", "expression"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


def test_discover_column_configs_values_are_classes() -> None:
    result = discover_column_configs()
    for cls in result.values():
        assert isinstance(cls, type)
        assert hasattr(cls, "model_fields")


# ---------------------------------------------------------------------------
# discover_sampler_types
# ---------------------------------------------------------------------------


def test_discover_sampler_types_returns_dict() -> None:
    result = discover_sampler_types()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_sampler_types_contains_expected_keys() -> None:
    result = discover_sampler_types()
    for expected_key in ("category", "uniform", "person"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


# ---------------------------------------------------------------------------
# discover_validator_types
# ---------------------------------------------------------------------------


def test_discover_validator_types_returns_dict() -> None:
    result = discover_validator_types()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_validator_types_contains_expected_keys() -> None:
    result = discover_validator_types()
    for expected_key in ("code", "remote"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


# ---------------------------------------------------------------------------
# discover_processor_configs
# ---------------------------------------------------------------------------


def test_discover_processor_configs_returns_dict() -> None:
    result = discover_processor_configs()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_processor_configs_contains_expected_keys() -> None:
    result = discover_processor_configs()
    assert "drop_columns" in result, f"Expected 'drop_columns' not found in {list(result.keys())}"


# ---------------------------------------------------------------------------
# discover_model_configs
# ---------------------------------------------------------------------------


def test_discover_model_configs_returns_dict() -> None:
    result = discover_model_configs()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_model_configs_contains_expected_keys() -> None:
    result = discover_model_configs()
    for expected_key in ("ModelConfig", "ChatCompletionInferenceParams"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


# ---------------------------------------------------------------------------
# discover_constraint_types
# ---------------------------------------------------------------------------


def test_discover_constraint_types_returns_dict() -> None:
    result = discover_constraint_types()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_constraint_types_contains_expected_keys() -> None:
    result = discover_constraint_types()
    assert "ScalarInequalityConstraint" in result


# ---------------------------------------------------------------------------
# discover_seed_types
# ---------------------------------------------------------------------------


def test_discover_seed_types_returns_dict() -> None:
    result = discover_seed_types()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_seed_types_contains_expected_keys() -> None:
    result = discover_seed_types()
    for expected_key in ("SeedConfig", "LocalFileSeedSource"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


# ---------------------------------------------------------------------------
# discover_mcp_types
# ---------------------------------------------------------------------------


def test_discover_mcp_types_returns_dict() -> None:
    result = discover_mcp_types()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_mcp_types_contains_expected_keys() -> None:
    result = discover_mcp_types()
    for expected_key in ("MCPProvider", "ToolConfig"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"
