# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import patch

from rich.table import Table

from data_designer.cli.commands.list import (
    display_mcp_providers,
    display_models,
    display_providers,
    display_tool_configs,
    list_command,
)
from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRepository
from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.repositories.provider_repository import ProviderRepository
from data_designer.cli.repositories.tool_repository import ToolRepository


@patch("data_designer.cli.commands.list.display_providers")
@patch("data_designer.cli.commands.list.display_models")
@patch("data_designer.cli.commands.list.display_mcp_providers")
@patch("data_designer.cli.commands.list.display_tool_configs")
def test_list_command(
    mock_display_tool_configs, mock_display_mcp_providers, mock_display_models, mock_display_providers
):
    """Test list command."""
    list_command()
    mock_display_providers.assert_called_once()
    mock_display_models.assert_called_once()
    mock_display_mcp_providers.assert_called_once()
    mock_display_tool_configs.assert_called_once()


@patch("data_designer.cli.ui.console.print")
def test_display_providers(mock_console_print, stub_provider_service):
    """Test display providers."""
    display_providers(stub_provider_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "Model Providers"


@patch("data_designer.cli.ui.console.print")
def test_display_models(mock_console_print, stub_model_service):
    """Test display models."""
    display_models(stub_model_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "Model Configurations"


@patch("data_designer.cli.commands.list.print_warning")
@patch("data_designer.cli.ui.console.print")
def test_display_providers_empty_registry(mock_console_print, mock_print_warning, tmp_path: Path):
    """Test display providers with empty registry."""
    repository = ProviderRepository(tmp_path)
    display_providers(repository)
    mock_print_warning.assert_called_once_with(
        "Providers have not been configured. Run 'data-designer config providers' to configure them."
    )
    mock_console_print.assert_called_once()


@patch("data_designer.cli.commands.list.print_error")
@patch("data_designer.cli.ui.console.print")
def test_display_providers_exception(mock_console_print, mock_print_error, stub_provider_service):
    """Test display providers with exception."""
    with patch.object(stub_provider_service.repository, "load", side_effect=Exception("Test error")):
        display_providers(stub_provider_service.repository)
    mock_print_error.assert_called_once_with("Error loading provider configuration: Test error")
    mock_console_print.assert_called_once()


@patch("data_designer.cli.commands.list.print_warning")
@patch("data_designer.cli.ui.console.print")
def test_display_models_empty_registry(mock_console_print, mock_print_warning, tmp_path: Path):
    """Test display models with empty registry."""
    repository = ModelRepository(tmp_path)
    display_models(repository)
    mock_print_warning.assert_called_once_with(
        "Models have not been configured. Run 'data-designer config models' to configure them."
    )
    mock_console_print.assert_called_once()


@patch("data_designer.cli.commands.list.print_error")
@patch("data_designer.cli.ui.console.print")
def test_display_models_exception(mock_console_print, mock_print_error, stub_model_service):
    """Test display models with exception."""
    with patch.object(stub_model_service.repository, "load", side_effect=Exception("Test error")):
        display_models(stub_model_service.repository)
    mock_print_error.assert_called_once_with("Error loading model configuration: Test error")
    mock_console_print.assert_called_once()


@patch("data_designer.cli.ui.console.print")
def test_display_mcp_providers(mock_console_print, stub_mcp_provider_service):
    """Test display MCP providers."""
    display_mcp_providers(stub_mcp_provider_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "MCP Providers"


@patch("data_designer.cli.commands.list.print_warning")
@patch("data_designer.cli.ui.console.print")
def test_display_mcp_providers_empty_registry(mock_console_print, mock_print_warning, tmp_path: Path):
    """Test display MCP providers with empty registry."""
    repository = MCPProviderRepository(tmp_path)
    display_mcp_providers(repository)
    mock_print_warning.assert_called_once_with(
        "MCP providers have not been configured. Run 'data-designer config mcp' to configure them."
    )
    mock_console_print.assert_called_once()


@patch("data_designer.cli.commands.list.print_error")
@patch("data_designer.cli.ui.console.print")
def test_display_mcp_providers_exception(mock_console_print, mock_print_error, stub_mcp_provider_service):
    """Test display MCP providers with exception."""
    with patch.object(stub_mcp_provider_service.repository, "load", side_effect=Exception("Test error")):
        display_mcp_providers(stub_mcp_provider_service.repository)
    mock_print_error.assert_called_once_with("Error loading MCP provider configuration: Test error")
    mock_console_print.assert_called_once()


@patch("data_designer.cli.ui.console.print")
def test_display_tool_configs(mock_console_print, stub_tool_service):
    """Test display tool configs."""
    display_tool_configs(stub_tool_service.repository)
    mock_console_print.call_count > 1
    assert isinstance(mock_console_print.call_args_list[0][0][0], Table)
    mock_console_print.call_args_list[0][0][0].title == "Tool Configurations"


@patch("data_designer.cli.commands.list.print_warning")
@patch("data_designer.cli.ui.console.print")
def test_display_tool_configs_empty_registry(mock_console_print, mock_print_warning, tmp_path: Path):
    """Test display tool configs with empty registry."""
    repository = ToolRepository(tmp_path)
    display_tool_configs(repository)
    mock_print_warning.assert_called_once_with(
        "Tool configs have not been configured. Run 'data-designer config tools' to configure them."
    )
    mock_console_print.assert_called_once()


@patch("data_designer.cli.commands.list.print_error")
@patch("data_designer.cli.ui.console.print")
def test_display_tool_configs_exception(mock_console_print, mock_print_error, stub_tool_service):
    """Test display tool configs with exception."""
    with patch.object(stub_tool_service.repository, "load", side_effect=Exception("Test error")):
        display_tool_configs(stub_tool_service.repository)
    mock_print_error.assert_called_once_with("Error loading tool configuration: Test error")
    mock_console_print.assert_called_once()
