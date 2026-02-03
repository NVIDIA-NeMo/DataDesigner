# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.tool_controller import ToolController
from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry
from data_designer.cli.repositories.tool_repository import ToolConfigRegistry
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider, ToolConfig

if TYPE_CHECKING:
    MCPProviderT = MCPProvider | LocalStdioMCPProvider


@pytest.fixture
def controller_with_mcp(tmp_path: Path, stub_mcp_providers: list[MCPProviderT]) -> ToolController:
    """Create a controller instance with MCP providers configured."""
    controller = ToolController(tmp_path)
    controller.mcp_provider_repository.save(MCPProviderRegistry(providers=stub_mcp_providers))
    return controller


@pytest.fixture
def controller_with_tools(
    tmp_path: Path,
    stub_mcp_providers: list[MCPProviderT],
    stub_tool_configs: list[ToolConfig],
) -> ToolController:
    """Create a controller instance with existing tool configs."""
    controller = ToolController(tmp_path)
    controller.mcp_provider_repository.save(MCPProviderRegistry(providers=stub_mcp_providers))
    controller.repository.save(ToolConfigRegistry(tool_configs=stub_tool_configs))
    return controller


def test_init(tmp_path: Path) -> None:
    """Test controller initialization sets up repositories and services correctly."""
    controller = ToolController(tmp_path)

    assert controller.config_dir == tmp_path
    assert controller.repository.config_dir == tmp_path
    assert controller.service.repository == controller.repository
    assert controller.mcp_provider_repository.config_dir == tmp_path
    assert controller.mcp_provider_service.repository == controller.mcp_provider_repository


@patch("data_designer.cli.controllers.tool_controller.print_error")
@patch("data_designer.cli.controllers.tool_controller.print_info")
@patch("data_designer.cli.controllers.tool_controller.print_header")
def test_run_with_no_mcp_providers(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_print_error: MagicMock,
    tmp_path: Path,
) -> None:
    """Test run exits early when no MCP providers are configured."""
    controller = ToolController(tmp_path)
    controller.run()

    mock_print_header.assert_called_once_with("Configure Tool Configs")
    mock_print_error.assert_called_once_with("No MCP providers available!")
    mock_print_info.assert_called_once_with("Please run 'data-designer config mcp' first to configure MCP providers.")


@patch("data_designer.cli.controllers.tool_controller.print_info")
@patch("data_designer.cli.controllers.tool_controller.print_header")
def test_run_with_no_tools_prompts_add(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test run with no existing tools defaults to add mode."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = None

    with patch(
        "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
        return_value=mock_builder,
    ):
        controller_with_mcp.run()

    mock_print_info.assert_any_call("No tool configs configured yet")
    mock_builder.run.assert_called_once()


def test_run_with_no_tools_adds_new_config(
    controller_with_mcp: ToolController,
    stub_new_tool_config: ToolConfig,
) -> None:
    """Test run with no existing tools successfully adds a new tool config."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = stub_new_tool_config

    with (
        patch(
            "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
            return_value=mock_builder,
        ),
        patch(
            "data_designer.cli.controllers.tool_controller.select_with_arrows",
            return_value="no",
        ),
    ):
        controller_with_mcp.run()

    configs = controller_with_mcp.service.list_all()
    assert len(configs) == 1
    assert configs[0].tool_alias == stub_new_tool_config.tool_alias


@patch("data_designer.cli.controllers.tool_controller.select_with_arrows", return_value="exit")
def test_run_with_existing_configs_and_exit(
    mock_select: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test run with existing configs respects exit choice."""
    initial_count = len(controller_with_tools.service.list_all())

    controller_with_tools.run()

    assert len(controller_with_tools.service.list_all()) == initial_count


@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_run_adds_multiple_configs(
    mock_select: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test run can add multiple tool configs sequentially."""
    config1 = ToolConfig(tool_alias="config-1", providers=["mcp-provider-1"])
    config2 = ToolConfig(tool_alias="config-2", providers=["mcp-provider-2"])

    mock_builder = MagicMock()
    mock_builder.run.side_effect = [config1, config2, None]
    mock_select.side_effect = ["yes", "no"]  # Add another? yes, then no

    with patch(
        "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
        return_value=mock_builder,
    ):
        controller_with_mcp.run()

    configs = controller_with_mcp.service.list_all()
    assert len(configs) == 2
    assert {c.tool_alias for c in configs} == {"config-1", "config-2"}


@patch("data_designer.cli.controllers.tool_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_run_deletes_config(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test run can delete a tool config through delete mode."""
    mock_select.side_effect = ["delete", "tool-config-1"]

    initial_count = len(controller_with_tools.service.list_all())
    controller_with_tools.run()

    remaining = controller_with_tools.service.list_all()
    assert len(remaining) == initial_count - 1
    assert "tool-config-1" not in [c.tool_alias for c in remaining]


@patch("data_designer.cli.controllers.tool_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows", return_value="delete_all")
def test_run_deletes_all_configs(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test run can delete all tool configs through delete_all mode."""
    controller_with_tools.run()

    assert len(controller_with_tools.service.list_all()) == 0


@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_run_updates_config(
    mock_select: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test run can update an existing tool config through update mode."""
    mock_select.side_effect = ["update", "tool-config-1"]

    updated_config = ToolConfig(
        tool_alias="tool-config-1-updated",
        providers=["mcp-provider-2"],
        max_tool_call_turns=20,
    )

    mock_builder = MagicMock()
    mock_builder.run.return_value = updated_config

    with patch(
        "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
        return_value=mock_builder,
    ):
        controller_with_tools.run()

    updated = controller_with_tools.service.get_by_alias("tool-config-1-updated")
    assert updated is not None
    assert updated.max_tool_call_turns == 20
    assert controller_with_tools.service.get_by_alias("tool-config-1") is None


@patch("data_designer.cli.controllers.tool_controller.print_error")
def test_handle_update_with_no_configs(
    mock_print_error: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test update mode with no configs shows error."""
    controller_with_mcp._handle_update([])

    mock_print_error.assert_called_once_with("No tool configs to update")


@patch("data_designer.cli.controllers.tool_controller.print_error")
def test_handle_delete_with_no_configs(
    mock_print_error: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test delete mode with no configs shows error."""
    controller_with_mcp._handle_delete([])

    mock_print_error.assert_called_once_with("No tool configs to delete")


def test_get_available_providers(controller_with_mcp: ToolController) -> None:
    """Test _get_available_providers returns provider names."""
    providers = controller_with_mcp._get_available_providers()

    assert "mcp-provider-1" in providers
    assert "mcp-provider-2" in providers
    assert "mcp-provider-stdio" in providers


def test_select_tool_config_displays_providers(
    controller_with_tools: ToolController,
    stub_tool_configs: list[ToolConfig],
) -> None:
    """Test _select_tool_config displays provider information."""
    with patch(
        "data_designer.cli.controllers.tool_controller.select_with_arrows",
        return_value=stub_tool_configs[0].tool_alias,
    ) as mock_select:
        result = controller_with_tools._select_tool_config(stub_tool_configs, "Select config")

    assert result == stub_tool_configs[0].tool_alias
    call_args = mock_select.call_args
    options = call_args[0][0]
    # Check that providers are shown in the option text
    expected_providers = ", ".join(stub_tool_configs[0].providers)
    assert f"(providers: {expected_providers})" in options[stub_tool_configs[0].tool_alias]


@patch("data_designer.cli.controllers.tool_controller.confirm_action", return_value=False)
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_delete_config_cancelled_by_user(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test delete is cancelled when user declines confirmation."""
    mock_select.side_effect = ["delete", "tool-config-1"]
    initial_count = len(controller_with_tools.service.list_all())

    controller_with_tools.run()

    assert len(controller_with_tools.service.list_all()) == initial_count


@patch("data_designer.cli.controllers.tool_controller.select_with_arrows", return_value=None)
def test_select_mode_returns_none_when_cancelled(
    mock_select: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test _select_mode returns None when user cancels."""
    result = controller_with_tools._select_mode()
    assert result is None


def test_confirm_add_another_yes(controller_with_mcp: ToolController) -> None:
    """Test _confirm_add_another returns True for yes."""
    with patch(
        "data_designer.cli.controllers.tool_controller.select_with_arrows",
        return_value="yes",
    ):
        assert controller_with_mcp._confirm_add_another() is True


def test_confirm_add_another_no(controller_with_mcp: ToolController) -> None:
    """Test _confirm_add_another returns False for no."""
    with patch(
        "data_designer.cli.controllers.tool_controller.select_with_arrows",
        return_value="no",
    ):
        assert controller_with_mcp._confirm_add_another() is False


@patch("data_designer.cli.controllers.tool_controller.print_error")
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_handle_update_config_not_found(
    mock_select: MagicMock,
    mock_print_error: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test update shows error when selected config not found."""
    mock_select.side_effect = ["update", "nonexistent-config"]

    controller_with_tools.run()

    mock_print_error.assert_called_with("Tool config 'nonexistent-config' not found")


@patch("data_designer.cli.controllers.tool_controller.print_error")
def test_handle_add_catches_value_error(
    mock_print_error: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test add handles ValueError from service."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = ToolConfig(tool_alias="test", providers=["mcp-provider-1"])

    available_providers = controller_with_mcp._get_available_providers()

    with (
        patch(
            "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
            return_value=mock_builder,
        ),
        patch.object(controller_with_mcp.service, "add", side_effect=ValueError("Duplicate alias")),
    ):
        controller_with_mcp._handle_add(available_providers)

    mock_print_error.assert_called_once_with("Failed to add tool config: Duplicate alias")


@patch("data_designer.cli.controllers.tool_controller.print_info")
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows", return_value=None)
def test_run_no_changes_made_on_none_mode(
    mock_select: MagicMock,
    mock_print_info: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test run prints 'No changes made' when mode is None."""
    controller_with_tools.run()

    mock_print_info.assert_any_call("No changes made")


@patch("data_designer.cli.controllers.tool_controller.print_error")
def test_handle_delete_all_with_no_configs(
    mock_print_error: MagicMock,
    controller_with_mcp: ToolController,
) -> None:
    """Test delete_all mode with no configs shows error."""
    controller_with_mcp._handle_delete_all([])

    mock_print_error.assert_called_once_with("No tool configs to delete")


@patch("data_designer.cli.controllers.tool_controller.print_error")
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows", return_value="tool-config-1")
def test_handle_update_catches_value_error(
    mock_select: MagicMock,
    mock_print_error: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test update handles ValueError from service."""
    updated_config = ToolConfig(tool_alias="updated", providers=["mcp-provider-1"])
    mock_builder = MagicMock()
    mock_builder.run.return_value = updated_config

    available_providers = controller_with_tools._get_available_providers()

    with (
        patch(
            "data_designer.cli.controllers.tool_controller.ToolFormBuilder",
            return_value=mock_builder,
        ),
        patch.object(controller_with_tools.service, "update", side_effect=ValueError("Update failed")),
    ):
        controller_with_tools._handle_update(available_providers)

    mock_print_error.assert_called_with("Failed to update tool config: Update failed")


@patch("data_designer.cli.controllers.tool_controller.print_error")
@patch("data_designer.cli.controllers.tool_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.tool_controller.select_with_arrows")
def test_handle_delete_catches_value_error(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    mock_print_error: MagicMock,
    controller_with_tools: ToolController,
) -> None:
    """Test delete handles ValueError from service."""
    mock_select.side_effect = ["delete", "tool-config-1"]

    available_providers = controller_with_tools._get_available_providers()

    with patch.object(controller_with_tools.service, "delete", side_effect=ValueError("Delete failed")):
        controller_with_tools._handle_delete(available_providers)

    mock_print_error.assert_called_with("Failed to delete tool config: Delete failed")
