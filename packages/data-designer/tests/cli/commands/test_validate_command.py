# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.validate import validate_command

# ---------------------------------------------------------------------------
# validate_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.controllers.generation_controller.GenerationController")
def test_validate_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """Test validate_command delegates to GenerationController.run_validate."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    validate_command(config_source="config.yaml")

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_validate.assert_called_once_with(config_source="config.yaml")


@patch("data_designer.cli.controllers.generation_controller.GenerationController")
def test_validate_command_passes_python_module_source(mock_ctrl_cls: MagicMock) -> None:
    """Test validate_command passes a .py source to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    validate_command(config_source="my_config.py")

    mock_ctrl.run_validate.assert_called_once_with(config_source="my_config.py")
