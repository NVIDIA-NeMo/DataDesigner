# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.cli.services.introspection.discovery import (
    _discover_by_modules,
    discover_column_configs,
    discover_constraint_types,
    discover_processor_configs,
    discover_sampler_types,
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
# _discover_by_modules
# ---------------------------------------------------------------------------


def test_discover_by_modules_returns_only_matching_modules() -> None:
    result = _discover_by_modules("models")
    import data_designer.config as dd

    lazy_imports: dict[str, tuple[str, str]] = getattr(dd, "_LAZY_IMPORTS", {})
    model_names = {name for name, (mod, _) in lazy_imports.items() if mod == "data_designer.config.models"}
    assert set(result.keys()) == model_names


def test_discover_by_modules_with_multiple_suffixes() -> None:
    result = _discover_by_modules("seed", "seed_source")
    assert "SeedConfig" in result
    assert "LocalFileSeedSource" in result


def test_discover_by_modules_unknown_suffix_returns_empty() -> None:
    result = _discover_by_modules("nonexistent_module")
    assert result == {}
