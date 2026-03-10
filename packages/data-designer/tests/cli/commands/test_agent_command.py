# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from data_designer.cli.controllers.agent_controller import AgentController
from data_designer.cli.main import app
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


def test_agent_context_command_outputs_json() -> None:
    runner = CliRunner()

    with patch("data_designer.cli.commands.agent.AgentController") as mock_controller:
        controller = MagicMock(spec=AgentController)
        controller.get_library_version.return_value = "1.2.3"
        controller.get_context.return_value = {
            "operations": [],
            "families": [],
            "types": {},
            "state": {},
            "builder": {},
        }
        mock_controller.return_value = controller

        result = runner.invoke(app, ["agent", "context"])

    assert result.exit_code == 0
    mock_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    controller.get_context.assert_called_once_with()
    payload = json.loads(result.output)
    assert payload == {
        "kind": "agent_context",
        "library_version": "1.2.3",
        "data": {"operations": [], "families": [], "types": {}, "state": {}, "builder": {}},
    }


def test_agent_schema_command_outputs_json_error() -> None:
    runner = CliRunner()

    with patch("data_designer.cli.commands.agent.AgentController") as mock_controller:
        controller = MagicMock(spec=AgentController)
        controller.get_library_version.return_value = "1.2.3"
        controller.get_schema.side_effect = ValueError("boom")
        mock_controller.return_value = controller

        result = runner.invoke(app, ["agent", "schema", "columns", "missing"])

    assert result.exit_code == 1
    payload = json.loads(result.output)
    assert payload == {
        "error": {
            "code": "internal_error",
            "message": "boom",
            "details": {"exception_type": "ValueError"},
        },
    }


def test_agent_state_model_aliases_command_outputs_json() -> None:
    runner = CliRunner()

    with patch("data_designer.cli.commands.agent.AgentController") as mock_controller:
        controller = MagicMock(spec=AgentController)
        controller.get_library_version.return_value = "1.2.3"
        controller.get_model_aliases_state.return_value = {"model_config_present": False, "items": []}
        mock_controller.return_value = controller

        result = runner.invoke(app, ["agent", "state", "model-aliases"])

    assert result.exit_code == 0
    mock_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    controller.get_model_aliases_state.assert_called_once_with()
    payload = json.loads(result.output)
    assert payload == {
        "kind": "agent_state_model_aliases",
        "library_version": "1.2.3",
        "data": {"model_config_present": False, "items": []},
    }
