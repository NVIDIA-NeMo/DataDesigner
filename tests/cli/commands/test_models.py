# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

from data_designer.cli.commands.models import models_command
from data_designer.cli.constants import DEFAULT_CONFIG_DIR
from data_designer.cli.controllers.model_controller import ModelController


@patch("data_designer.cli.commands.models.ModelController")
def test_models_command(mock_model_controller):
    mock_model_controller_instance = MagicMock(spec=ModelController)
    mock_model_controller.return_value = mock_model_controller_instance
    models_command(config_dir=None)
    mock_model_controller.assert_called_once()
    mock_model_controller.call_args[0][0] == DEFAULT_CONFIG_DIR
    mock_model_controller_instance.run.assert_called_once()


@patch("data_designer.cli.commands.models.ModelController")
def test_models_command_with_config_dir(mock_model_controller, tmp_path: Path):
    mock_model_controller_instance = MagicMock(spec=ModelController)
    mock_model_controller.return_value = mock_model_controller_instance
    models_command(config_dir=str(tmp_path))
    mock_model_controller.assert_called_once()
    mock_model_controller.call_args[0][0] == str(tmp_path)
    mock_model_controller_instance.run.assert_called_once()
