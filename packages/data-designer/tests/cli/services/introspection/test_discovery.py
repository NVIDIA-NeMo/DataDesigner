# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest

from data_designer.cli.services.introspection.discovery import (
    _discover_by_modules,
    discover_column_configs,
    discover_constraint_types,
    discover_importable_names,
    discover_interface_classes,
    discover_mcp_types,
    discover_model_configs,
    discover_namespace_tree,
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


# ---------------------------------------------------------------------------
# discover_namespace_tree
# ---------------------------------------------------------------------------


def test_discover_namespace_tree_returns_paths_and_tree() -> None:
    result = discover_namespace_tree()
    assert "paths" in result
    assert "tree" in result


def test_discover_namespace_tree_paths_non_empty() -> None:
    result = discover_namespace_tree()
    assert isinstance(result["paths"], list)
    assert len(result["paths"]) > 0
    for p in result["paths"]:
        assert isinstance(p, str)


def test_discover_namespace_tree_root_is_data_designer() -> None:
    result = discover_namespace_tree()
    tree = result["tree"]
    assert tree["name"] == "data_designer"
    assert tree["is_package"] is True


def test_discover_namespace_tree_contains_expected_children() -> None:
    result = discover_namespace_tree()
    child_names = [c["name"] for c in result["tree"]["children"]]
    for expected in ("config", "engine", "cli"):
        assert expected in child_names, f"Expected '{expected}' in {child_names}"


def test_discover_namespace_tree_children_have_correct_structure() -> None:
    result = discover_namespace_tree()
    for child in result["tree"]["children"]:
        assert "name" in child
        assert "is_package" in child
        assert "children" in child
        assert isinstance(child["name"], str)
        assert isinstance(child["is_package"], bool)
        assert isinstance(child["children"], list)


def test_discover_namespace_tree_negative_depth_raises() -> None:
    """Invalid max_depth < 0 raises ValueError with actionable message."""
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        discover_namespace_tree(max_depth=-1)


def test_discover_namespace_tree_import_errors_structure() -> None:
    """When present, import_errors is a list of dicts with module and message."""
    result = discover_namespace_tree()
    if "import_errors" in result:
        errors = result["import_errors"]
        assert isinstance(errors, list)
        for err in errors:
            assert "module" in err
            assert "message" in err
            assert isinstance(err["module"], str)
            assert isinstance(err["message"], str)


# ---------------------------------------------------------------------------
# discover_interface_classes
# ---------------------------------------------------------------------------


def test_discover_interface_classes_returns_dict() -> None:
    result = discover_interface_classes()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_interface_classes_contains_expected_keys() -> None:
    result = discover_interface_classes()
    for expected_key in ("DataDesigner", "DatasetCreationResults", "PreviewResults", "RunConfig"):
        assert expected_key in result, f"Expected key '{expected_key}' not found in {list(result.keys())}"


def test_discover_interface_classes_values_are_classes() -> None:
    result = discover_interface_classes()
    for cls in result.values():
        assert isinstance(cls, type)


# ---------------------------------------------------------------------------
# discover_importable_names
# ---------------------------------------------------------------------------


def test_discover_importable_names_returns_dict() -> None:
    result = discover_importable_names()
    assert isinstance(result, dict)
    assert len(result) > 0


def test_discover_importable_names_has_column_configs_category() -> None:
    result = discover_importable_names()
    assert "Column Configs" in result, f"Expected 'Column Configs' in {list(result.keys())}"


def test_discover_importable_names_has_interface_category() -> None:
    result = discover_importable_names()
    assert "Interface" in result, f"Expected 'Interface' in {list(result.keys())}"


def test_discover_importable_names_entries_have_name_and_module() -> None:
    result = discover_importable_names()
    for category, entries in result.items():
        assert isinstance(entries, list), f"Category '{category}' value is not a list"
        for entry in entries:
            assert "name" in entry, f"Entry in '{category}' missing 'name'"
            assert "module" in entry, f"Entry in '{category}' missing 'module'"


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


# ---------------------------------------------------------------------------
# discover_interface_classes — error class exclusion
# ---------------------------------------------------------------------------


def test_discover_interface_classes_excludes_exceptions() -> None:
    result = discover_interface_classes()
    for name, cls in result.items():
        assert not issubclass(cls, Exception), f"{name} is an Exception subclass and should be excluded"
