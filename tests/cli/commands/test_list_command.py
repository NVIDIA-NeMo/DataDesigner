# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from rich.table import Table

from data_designer.cli.commands.list import display_models, display_providers, list_command
from data_designer.cli.constants import DEFAULT_CONFIG_DIR


@patch("data_designer.cli.commands.list.display_providers")
@patch("data_designer.cli.commands.list.display_models")
def test_list_command(mock_display_models, mock_display_providers):
    """Test list command."""
    list_command(config_dir=None)
    mock_display_providers.assert_called_once()
    mock_display_providers.call_args[0][0].config_dir == DEFAULT_CONFIG_DIR
    mock_display_models.assert_called_once()
    mock_display_models.call_args[0][0].config_dir == DEFAULT_CONFIG_DIR


@patch("data_designer.cli.commands.list.console.print")
def test_display_providers(mock_console_print, stub_provider_service):
    """Test display providers."""
    display_providers(stub_provider_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "Model Providers"


@patch("data_designer.cli.commands.list.console.print")
def test_display_models(mock_console_print, stub_model_service):
    """Test display models."""
    display_models(stub_model_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "Model Configurations"
