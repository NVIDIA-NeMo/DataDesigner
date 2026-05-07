# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.validate import validate_command

# ---------------------------------------------------------------------------
# validate_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.commands.validate.GenerationController")
def test_validate_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """Test validate_command delegates to GenerationController.run_validate."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    validate_command(workflow_args=["config.yaml"])

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_validate.assert_called_once_with(config_source="config.yaml", recipe=None, workflow_args=())


@patch("data_designer.cli.commands.validate.GenerationController")
def test_validate_command_passes_python_module_source(mock_ctrl_cls: MagicMock) -> None:
    """Test validate_command passes a .py source to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    validate_command(workflow_args=["my_config.py", "--seed-path", "seed.jsonl"])

    mock_ctrl.run_validate.assert_called_once_with(
        config_source="my_config.py",
        recipe=None,
        workflow_args=("--seed-path", "seed.jsonl"),
    )


@patch("data_designer.cli.commands.validate.GenerationController")
def test_validate_command_passes_recipe_target(mock_ctrl_cls: MagicMock) -> None:
    """Test validate_command forwards --recipe and workflow args to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    validate_command(workflow_args=["--input-dir", "docs"], recipe="retrieval-sdg")

    mock_ctrl.run_validate.assert_called_once_with(
        config_source=None,
        recipe="retrieval-sdg",
        workflow_args=("--input-dir", "docs"),
    )
