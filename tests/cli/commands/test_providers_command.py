# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

from data_designer.cli.commands.providers import providers_command
from data_designer.cli.constants import DEFAULT_CONFIG_DIR
from data_designer.cli.controllers.provider_controller import ProviderController


@patch("data_designer.cli.commands.providers.ProviderController")
def test_providers_command(mock_provider_controller):
    mock_provider_controller_instance = MagicMock(spec=ProviderController)
    mock_provider_controller.return_value = mock_provider_controller_instance
    providers_command(config_dir=None)
    mock_provider_controller.assert_called_once()
    mock_provider_controller.call_args[0][0] == DEFAULT_CONFIG_DIR
    mock_provider_controller_instance.run.assert_called_once()


@patch("data_designer.cli.commands.providers.ProviderController")
def test_providers_command_with_config_dir(mock_provider_controller, tmp_path: Path):
    mock_provider_controller_instance = MagicMock(spec=ProviderController)
    mock_provider_controller.return_value = mock_provider_controller_instance
    providers_command(config_dir=str(tmp_path))
    mock_provider_controller.assert_called_once()
    mock_provider_controller.call_args[0][0] == str(tmp_path)
    mock_provider_controller_instance.run.assert_called_once()
