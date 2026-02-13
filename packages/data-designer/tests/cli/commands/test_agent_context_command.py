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
    result = runner.invoke(app, ["agent-context", "--help"])
    assert result.exit_code == 0
    assert "columns" in result.output


# ---------------------------------------------------------------------------
# columns
# ---------------------------------------------------------------------------


def test_columns_list() -> None:
    result = runner.invoke(app, ["agent-context", "columns", "--list"])
    assert result.exit_code == 0
    assert "llm-text" in result.output


def test_columns_specific_type() -> None:
    result = runner.invoke(app, ["agent-context", "columns", "llm-text"])
    assert result.exit_code == 0
    assert "LLMTextColumnConfig" in result.output


def test_columns_json_format() -> None:
    result = runner.invoke(app, ["agent-context", "columns", "llm-text", "--format", "json"])
    assert result.exit_code == 0
    data = json.loads(result.output)
    assert isinstance(data, dict)
    assert data["class_name"] == "LLMTextColumnConfig"


def test_columns_nonexistent_exits_with_error() -> None:
    result = runner.invoke(app, ["agent-context", "columns", "nonexistent"])
    assert result.exit_code == 1


# ---------------------------------------------------------------------------
# samplers
# ---------------------------------------------------------------------------


def test_samplers_specific() -> None:
    result = runner.invoke(app, ["agent-context", "samplers", "category"])
    assert result.exit_code == 0
    assert "CATEGORY" in result.output


def test_samplers_list() -> None:
    result = runner.invoke(app, ["agent-context", "samplers", "--list"])
    assert result.exit_code == 0
    assert "category" in result.output


# ---------------------------------------------------------------------------
# overview
# ---------------------------------------------------------------------------


def test_overview() -> None:
    result = runner.invoke(app, ["agent-context", "overview"])
    assert result.exit_code == 0
    assert "Type Counts" in result.output


# ---------------------------------------------------------------------------
# builder
# ---------------------------------------------------------------------------


def test_builder() -> None:
    result = runner.invoke(app, ["agent-context", "builder"])
    assert result.exit_code == 0
    assert "add_column" in result.output


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


def test_models() -> None:
    result = runner.invoke(app, ["agent-context", "models"])
    assert result.exit_code == 0
    assert "ModelConfig" in result.output


# ---------------------------------------------------------------------------
# constraints
# ---------------------------------------------------------------------------


def test_constraints() -> None:
    result = runner.invoke(app, ["agent-context", "constraints"])
    assert result.exit_code == 0
    output = result.output
    assert "ScalarInequalityConstraint" in output or "InequalityOperator" in output


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------


def test_seeds() -> None:
    result = runner.invoke(app, ["agent-context", "seeds"])
    assert result.exit_code == 0
    assert "SeedConfig" in result.output


# ---------------------------------------------------------------------------
# mcp
# ---------------------------------------------------------------------------


def test_mcp() -> None:
    result = runner.invoke(app, ["agent-context", "mcp"])
    assert result.exit_code == 0
    assert "ToolConfig" in result.output
