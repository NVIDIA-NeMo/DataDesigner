# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from data_designer.cli.main import app

_PATCH = "data_designer.cli.commands.agent"


@pytest.mark.parametrize(
    "args,data_fn,format_fn,expected_text",
    [
        (["agent", "context"], "get_context", "format_context_text", "Data Designer"),
        (["agent", "types", "columns"], "get_types", "format_types_text", "columns"),
        (["agent", "builder"], "get_builder_api", "format_builder_text", "Builder:"),
        (["agent", "state", "model-aliases"], "get_model_aliases_state", "format_model_aliases_text", "model aliases"),
        (
            ["agent", "state", "persona-datasets"],
            "get_persona_datasets_state",
            "format_persona_datasets_text",
            "persona",
        ),
    ],
    ids=["context", "types", "builder", "model-aliases", "persona-datasets"],
)
def test_commands_default_text_mode(args: list[str], data_fn: str, format_fn: str, expected_text: str) -> None:
    runner = CliRunner()
    with (
        patch(f"{_PATCH}.{data_fn}", return_value={"stub": True}) as mock_get,
        patch(f"{_PATCH}.{format_fn}", return_value=expected_text),
    ):
        result = runner.invoke(app, args)

    assert result.exit_code == 0
    assert expected_text in result.output
    mock_get.assert_called_once()


def test_schema_command_default_outputs_text() -> None:
    runner = CliRunner()
    with (
        patch(f"{_PATCH}.get_schema", return_value={"type_name": "llm-text", "schema": {}}),
        patch(f"{_PATCH}.format_schema_text", return_value="# llm-text\n{}"),
    ):
        result = runner.invoke(app, ["agent", "schema", "columns", "llm-text"])

    assert result.exit_code == 0
    assert "llm-text" in result.output


def test_error_outputs_message_to_stderr() -> None:
    runner = CliRunner()
    with patch(f"{_PATCH}.get_schema", side_effect=ValueError("boom")):
        result = runner.invoke(app, ["agent", "schema", "columns", "missing"])

    assert result.exit_code == 1
    assert result.stdout == ""
    assert "internal_error" in result.stderr
    assert "boom" in result.stderr
