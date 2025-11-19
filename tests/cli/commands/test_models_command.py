# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.models import models_command
from data_designer.cli.controllers.model_controller import ModelController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


@patch("data_designer.cli.commands.models.ModelController")
def test_models_command(mock_model_controller):
    mock_model_controller_instance = MagicMock(spec=ModelController)
    mock_model_controller.return_value = mock_model_controller_instance
    models_command()
    mock_model_controller.assert_called_once()
    mock_model_controller.call_args[0][0] == DATA_DESIGNER_HOME
    mock_model_controller_instance.run.assert_called_once()
