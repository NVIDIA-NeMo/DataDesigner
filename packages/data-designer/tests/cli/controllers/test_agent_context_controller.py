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
