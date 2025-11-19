# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.providers import providers_command
from data_designer.cli.controllers.provider_controller import ProviderController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


@patch("data_designer.cli.commands.providers.ProviderController")
def test_providers_command(mock_provider_controller):
    mock_provider_controller_instance = MagicMock(spec=ProviderController)
    mock_provider_controller.return_value = mock_provider_controller_instance
    providers_command()
    mock_provider_controller.assert_called_once()
    mock_provider_controller.call_args[0][0] == DATA_DESIGNER_HOME
    mock_provider_controller_instance.run.assert_called_once()
