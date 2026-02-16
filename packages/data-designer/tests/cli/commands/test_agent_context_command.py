# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# help
# ---------------------------------------------------------------------------


def test_agent_context_help() -> None:
    result = runner.invoke(app, ["introspect", "--help"])
    assert result.exit_code == 0
    assert "columns" in result.output


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------


def test_columns_list() -> None:
    result = runner.invoke(app, ["introspect", "columns", "--list"])
    assert result.exit_code == 0
    assert "llm-text" in result.output


def test_columns_specific_type() -> None:
    result = runner.invoke(app, ["introspect", "columns", "llm-text"])
    assert result.exit_code == 0
    assert "LLMTextColumnConfig" in result.output


def test_columns_json_format() -> None:
    result = runner.invoke(app, ["introspect", "columns", "llm-text", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert data["class_name"] == "LLMTextColumnConfig"


def test_columns_nonexistent_exits_with_error() -> None:
    result = runner.invoke(app, ["introspect", "columns", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def test_samplers_specific() -> None:
    result = runner.invoke(app, ["introspect", "samplers", "category"])
    assert result.exit_code == 0
    assert "CATEGORY" in result.output


def test_samplers_list() -> None:
    result = runner.invoke(app, ["introspect", "samplers", "--list"])
    assert result.exit_code == 0
    assert "category" in result.output


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------


def test_overview() -> None:
    result = runner.invoke(app, ["introspect", "overview"])
    assert result.exit_code == 0
    assert "Type Counts" in result.output


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_builder() -> None:
    result = runner.invoke(app, ["introspect", "builder"])
    assert result.exit_code == 0
    assert "add_column" in result.output
    assert "DataDesignerConfigBuilder" in result.output
    assert "Parameters:" in result.output


def test_builder_json() -> None:
    result = runner.invoke(app, ["introspect", "builder", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    method_names = [m["name"] for m in data]
    assert "add_column" in method_names


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models() -> None:
    result = runner.invoke(app, ["introspect", "models"])
    assert result.exit_code == 0
    assert "ModelConfig" in result.output
    assert "description:" in result.output


def test_models_json() -> None:
    result = runner.invoke(app, ["introspect", "models", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    class_names = [item.get("class_name", "") for item in data if isinstance(item, dict)]
    assert "ModelConfig" in class_names


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def test_constraints() -> None:
    result = runner.invoke(app, ["introspect", "constraints"])
    assert result.exit_code == 0
    output = result.output
    assert "ScalarInequalityConstraint" in output or "InequalityOperator" in output


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def test_seeds() -> None:
    result = runner.invoke(app, ["introspect", "seeds"])
    assert result.exit_code == 0
    assert "SeedConfig" in result.output
    assert "SamplingStrategy" in result.output


def test_seeds_json() -> None:
    result = runner.invoke(app, ["introspect", "seeds", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    class_names = [item.get("class_name", "") for item in data if isinstance(item, dict)]
    assert "SeedConfig" in class_names


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


def test_mcp() -> None:
    result = runner.invoke(app, ["introspect", "mcp"])
    assert result.exit_code == 0
    assert "ToolConfig" in result.output
    assert "MCPProvider" in result.output or "LocalStdioMCPProvider" in result.output


def test_mcp_json() -> None:
    result = runner.invoke(app, ["introspect", "mcp", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, list)
    class_names = [item.get("class_name", "") for item in data if isinstance(item, dict)]
    assert "ToolConfig" in class_names


# ---------------------------------------------------------------------------
# code-structure
# ---------------------------------------------------------------------------


def test_code_structure() -> None:
    result = runner.invoke(app, ["introspect", "code-structure"])
    assert result.exit_code == 0
    assert "data_designer code structure" in result.output
    assert "├──" in result.output


def test_code_structure_shows_subpackages() -> None:
    result = runner.invoke(app, ["introspect", "code-structure"])
    assert result.exit_code == 0
    for pkg in ("config/", "engine/", "cli/"):
        assert pkg in result.output, f"Expected '{pkg}' in output"


def test_code_structure_json_format() -> None:
    result = runner.invoke(app, ["introspect", "code-structure", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "paths" in data
    assert "tree" in data


def test_code_structure_shows_agent_guidance() -> None:
    result = runner.invoke(app, ["introspect", "code-structure"])
    assert result.exit_code == 0
    assert "Only read source files directly" in result.output


# ---------------------------------------------------------------------------
# interface
# ---------------------------------------------------------------------------


def test_interface() -> None:
    result = runner.invoke(app, ["introspect", "interface"])
    assert result.exit_code == 0
    assert "DataDesigner" in result.output
    assert "create" in result.output


def test_interface_json() -> None:
    result = runner.invoke(app, ["introspect", "interface", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert "methods" in data
    assert "schemas" in data


def test_interface_shows_result_types() -> None:
    result = runner.invoke(app, ["introspect", "interface"])
    assert result.exit_code == 0
    assert "DatasetCreationResults" in result.output


# ---------------------------------------------------------------------------
# imports
# ---------------------------------------------------------------------------


def test_imports() -> None:
    result = runner.invoke(app, ["introspect", "imports"])
    assert result.exit_code == 0
    assert "from data_designer.config import" in result.output


def test_imports_json() -> None:
    result = runner.invoke(app, ["introspect", "imports", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert len(data) > 0


# ---------------------------------------------------------------------------
# format validation
# ---------------------------------------------------------------------------


def test_invalid_format_rejected() -> None:
    result = runner.invoke(app, ["introspect", "columns", "--list", "--format", "xml"])
    assert result.exit_code != 0


def test_invalid_format_rejected_on_builder() -> None:
    result = runner.invoke(app, ["introspect", "builder", "--format", "yaml"])
    assert result.exit_code != 0


def test_valid_format_text() -> None:
    result = runner.invoke(app, ["introspect", "columns", "--list", "--format", "text"])
    assert result.exit_code == 0


def test_valid_format_json() -> None:
    result = runner.invoke(app, ["introspect", "columns", "--list", "--format", "json"])
    assert result.exit_code == 0
