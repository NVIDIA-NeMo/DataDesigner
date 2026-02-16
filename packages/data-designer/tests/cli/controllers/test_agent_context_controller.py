# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import click.exceptions
import pytest

from data_designer.cli.controllers.agent_context_controller import AgentContextController

# ---------------------------------------------------------------------------
# show_columns
# ---------------------------------------------------------------------------


def test_show_columns_list_mode(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_columns(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    assert "llm-text" in captured.out
    assert "sampler" in captured.out


def test_show_columns_specific_type(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_columns(type_name="llm-text", list_mode=False)
    captured = capsys.readouterr()
    assert "LLMTextColumnConfig" in captured.out


def test_show_columns_all(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_columns(type_name="all", list_mode=False)
    captured = capsys.readouterr()
    assert "llm-text" in captured.out
    assert "sampler" in captured.out


def test_show_columns_nonexistent_type_exits() -> None:
    controller = AgentContextController(output_format="text")
    with pytest.raises(click.exceptions.Exit):
        controller.show_columns(type_name="nonexistent_type_xyz", list_mode=False)


def test_show_columns_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_columns(type_name="llm-text", list_mode=False)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert data["class_name"] == "LLMTextColumnConfig"


def test_show_columns_list_json_format(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_columns(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert "llm-text" in data


# ---------------------------------------------------------------------------
# show_overview
# ---------------------------------------------------------------------------


def test_show_overview_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_overview()
    captured = capsys.readouterr()
    assert "Data Designer API Overview" in captured.out
    assert "Type Counts:" in captured.out


def test_show_overview_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_overview()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "type_counts" in data
    assert "builder_methods" in data
    assert isinstance(data["type_counts"], dict)
    assert len(data["type_counts"]) > 0
    assert isinstance(data["builder_methods"], list)
    assert len(data["builder_methods"]) > 0


# ---------------------------------------------------------------------------
# show_samplers
# ---------------------------------------------------------------------------


def test_show_samplers_list(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_samplers(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    assert "category" in captured.out


def test_show_samplers_specific(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_samplers(type_name="category", list_mode=False)
    captured = capsys.readouterr()
    assert "CATEGORY" in captured.out


# ---------------------------------------------------------------------------
# show_models
# ---------------------------------------------------------------------------


def test_show_models(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_models()
    captured = capsys.readouterr()
    assert "ModelConfig" in captured.out


# ---------------------------------------------------------------------------
# show_builder
# ---------------------------------------------------------------------------


def test_show_builder(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_builder()
    captured = capsys.readouterr()
    assert "add_column" in captured.out


# ---------------------------------------------------------------------------
# show_constraints
# ---------------------------------------------------------------------------


def test_show_constraints(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_constraints()
    captured = capsys.readouterr()
    assert "ScalarInequalityConstraint" in captured.out


# ---------------------------------------------------------------------------
# show_seeds
# ---------------------------------------------------------------------------


def test_show_seeds(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_seeds()
    captured = capsys.readouterr()
    assert "SeedConfig" in captured.out


# ---------------------------------------------------------------------------
# show_mcp
# ---------------------------------------------------------------------------


def test_show_mcp(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_mcp()
    captured = capsys.readouterr()
    assert "ToolConfig" in captured.out


# ---------------------------------------------------------------------------
# show_interface
# ---------------------------------------------------------------------------


def test_show_interface_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_interface()
    captured = capsys.readouterr()
    assert "DataDesigner" in captured.out
    assert "DatasetCreationResults" in captured.out
    assert "RunConfig" in captured.out


def test_show_interface_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_interface()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "methods" in data
    assert "schemas" in data
    assert "DataDesigner" in data["methods"]
    assert isinstance(data["schemas"], list)
    assert len(data["schemas"]) > 0


# ---------------------------------------------------------------------------
# show_validators
# ---------------------------------------------------------------------------


def test_show_validators_list_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_validators(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    assert "validator_type" in captured.out
    assert "params_class" in captured.out


def test_show_validators_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_validators(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert len(data) > 0


def test_show_validators_specific_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_validators(type_name="code", list_mode=False)
    captured = capsys.readouterr()
    assert "CODE" in captured.out


def test_show_validators_specific_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_validators(type_name="code", list_mode=False)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert "fields" in data


# ---------------------------------------------------------------------------
# show_processors
# ---------------------------------------------------------------------------


def test_show_processors_list_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_processors(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    assert "processor_type" in captured.out
    assert "config_class" in captured.out


def test_show_processors_list_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_processors(type_name=None, list_mode=True)
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert len(data) > 0


# ---------------------------------------------------------------------------
# show_imports (with category filter)
# ---------------------------------------------------------------------------


def test_show_imports_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_imports()
    captured = capsys.readouterr()
    assert "from data_designer.config import" in captured.out


def test_show_imports_with_category_filter(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_imports(category="columns")
    captured = capsys.readouterr()
    assert "Column Configs" in captured.out
    assert "from data_designer.config import" in captured.out


def test_show_imports_with_category_filter_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_imports(category="columns")
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert isinstance(data, dict)
    assert "Column Configs" in data


def test_show_imports_with_invalid_category(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    with pytest.raises(click.exceptions.Exit):
        controller.show_imports(category="nonexistent_xyz")


# ---------------------------------------------------------------------------
# show_code_structure
# ---------------------------------------------------------------------------


def test_show_code_structure_text(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="text")
    controller.show_code_structure()
    captured = capsys.readouterr()
    assert "data_designer code structure" in captured.out
    assert "data_designer/" in captured.out


def test_show_code_structure_json(capsys: pytest.CaptureFixture[str]) -> None:
    controller = AgentContextController(output_format="json")
    controller.show_code_structure()
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "paths" in data
    assert "tree" in data
    assert data["tree"]["name"] == "data_designer"


# ---------------------------------------------------------------------------
# _match_category
# ---------------------------------------------------------------------------


def test_match_category_exact_match() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    assert AgentContextController._match_category("Column Configs", keys) == "Column Configs"


def test_match_category_exact_match_case_insensitive() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    assert AgentContextController._match_category("column configs", keys) == "Column Configs"
    assert AgentContextController._match_category("BUILDER", keys) == "Builder"


def test_match_category_first_word_stem_match() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # "columns" -> rstrip("s") -> "column", matches first word "column" of "Column Configs"
    assert AgentContextController._match_category("columns", keys) == "Column Configs"


def test_match_category_first_word_stem_match_singular() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # "column" is already stemmed, matches first word "column"
    assert AgentContextController._match_category("column", keys) == "Column Configs"


def test_match_category_any_word_stem_match() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # "configs" -> rstrip("s") -> "config", matches second word of "Column Configs"
    assert AgentContextController._match_category("configs", keys) == "Column Configs"


def test_match_category_substring_match() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # "uild" is a substring of "Builder"
    assert AgentContextController._match_category("uild", keys) == "Builder"


def test_match_category_substring_picks_earliest_position() -> None:
    keys = ["ABC-foo", "foo-ABC"]
    # "foo" appears at position 4 in "ABC-foo" and position 0 in "foo-ABC"
    assert AgentContextController._match_category("foo", keys) == "foo-ABC"


def test_match_category_no_match() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    assert AgentContextController._match_category("zzzzz_nonexistent", keys) is None


def test_match_category_empty_string() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # Empty string is a substring of everything; earliest position (0) wins
    result = AgentContextController._match_category("", keys)
    assert result is not None


def test_match_category_process_rstrip_s_edge_case() -> None:
    """Words ending in 's' naturally (like 'process') still work after rstrip('s')."""
    keys = ["Processors", "Builder"]
    # "process" -> rstrip("s") -> "proces"
    # First-word stem: "Processors" first word is "processor" -> rstrip("s") -> "processor" != "proces"
    # Any-word stem: same
    # Falls to substring: "process" is a substring of "Processors" at pos 0
    assert AgentContextController._match_category("process", keys) == "Processors"


def test_match_category_empty_keys_list() -> None:
    assert AgentContextController._match_category("anything", []) is None


def test_match_category_model_stem() -> None:
    keys = ["Column Configs", "Builder", "Model Configs"]
    # "models" -> rstrip("s") -> "model", matches first word "model" of "Model Configs"
    assert AgentContextController._match_category("models", keys) == "Model Configs"
