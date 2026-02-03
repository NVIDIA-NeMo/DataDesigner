# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.mcp_provider_controller import MCPProviderController
from data_designer.cli.repositories.mcp_provider_repository import MCPProviderRegistry
from data_designer.config.mcp import LocalStdioMCPProvider, MCPProvider

if TYPE_CHECKING:
    MCPProviderT = MCPProvider | LocalStdioMCPProvider


@pytest.fixture
def controller(tmp_path: Path) -> MCPProviderController:
    """Create a controller instance for testing."""
    return MCPProviderController(tmp_path)


@pytest.fixture
def controller_with_providers(tmp_path: Path, stub_mcp_providers: list[MCPProviderT]) -> MCPProviderController:
    """Create a controller instance with existing MCP providers."""
    controller = MCPProviderController(tmp_path)
    controller.repository.save(MCPProviderRegistry(providers=stub_mcp_providers))
    return controller


def test_init(tmp_path: Path) -> None:
    """Test controller initialization sets up repository and service correctly."""
    controller = MCPProviderController(tmp_path)

    assert controller.config_dir == tmp_path
    assert controller.repository.config_dir == tmp_path
    assert controller.service.repository == controller.repository


@patch("data_designer.cli.controllers.mcp_provider_controller.print_info")
@patch("data_designer.cli.controllers.mcp_provider_controller.print_header")
def test_run_with_no_providers_prompts_add(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    controller: MCPProviderController,
) -> None:
    """Test run with no existing providers defaults to add mode."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = None

    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.MCPProviderFormBuilder",
        return_value=mock_builder,
    ):
        controller.run()

    mock_print_header.assert_called_once_with("Configure MCP Providers")
    mock_print_info.assert_any_call("No MCP providers configured yet")
    mock_builder.run.assert_called_once()


def test_run_with_no_providers_adds_new_provider(
    controller: MCPProviderController,
    stub_new_mcp_provider: MCPProvider,
) -> None:
    """Test run with no existing providers successfully adds a new provider."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = stub_new_mcp_provider

    with (
        patch(
            "data_designer.cli.controllers.mcp_provider_controller.MCPProviderFormBuilder",
            return_value=mock_builder,
        ),
        patch(
            "data_designer.cli.controllers.mcp_provider_controller.select_with_arrows",
            return_value="no",
        ),
    ):
        controller.run()

    providers = controller.service.list_all()
    assert len(providers) == 1
    assert providers[0].name == stub_new_mcp_provider.name


@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows", return_value="exit")
def test_run_with_existing_providers_and_exit(
    mock_select: MagicMock,
    controller_with_providers: MCPProviderController,
    stub_mcp_providers: list[MCPProviderT],
) -> None:
    """Test run with existing providers respects exit choice."""
    initial_count = len(controller_with_providers.service.list_all())

    controller_with_providers.run()

    assert len(controller_with_providers.service.list_all()) == initial_count


@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_run_adds_multiple_providers(
    mock_select: MagicMock,
    controller: MCPProviderController,
) -> None:
    """Test run can add multiple providers sequentially."""
    provider1 = MCPProvider(
        name="provider-1",
        endpoint="http://localhost:8080/sse",
        api_key="key1",
    )
    provider2 = MCPProvider(
        name="provider-2",
        endpoint="http://localhost:8081/sse",
        api_key="key2",
    )

    mock_builder = MagicMock()
    mock_builder.run.side_effect = [provider1, provider2, None]
    mock_select.side_effect = ["yes", "no"]  # Add another? yes, then no

    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.MCPProviderFormBuilder",
        return_value=mock_builder,
    ):
        controller.run()

    providers = controller.service.list_all()
    assert len(providers) == 2
    assert {p.name for p in providers} == {"provider-1", "provider-2"}


@patch("data_designer.cli.controllers.mcp_provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_run_deletes_provider(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test run can delete a provider through delete mode."""
    mock_select.side_effect = ["delete", "mcp-provider-1"]

    initial_count = len(controller_with_providers.service.list_all())
    controller_with_providers.run()

    remaining = controller_with_providers.service.list_all()
    assert len(remaining) == initial_count - 1
    assert "mcp-provider-1" not in [p.name for p in remaining]


@patch("data_designer.cli.controllers.mcp_provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows", return_value="delete_all")
def test_run_deletes_all_providers(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test run can delete all providers through delete_all mode."""
    controller_with_providers.run()

    assert len(controller_with_providers.service.list_all()) == 0


@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_run_updates_provider(
    mock_select: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test run can update an existing provider through update mode."""
    mock_select.side_effect = ["update", "mcp-provider-1"]

    updated_provider = MCPProvider(
        name="mcp-provider-1-updated",
        endpoint="http://localhost:9999/sse",
        api_key="new-key",
    )

    mock_builder = MagicMock()
    mock_builder.run.return_value = updated_provider

    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.MCPProviderFormBuilder",
        return_value=mock_builder,
    ):
        controller_with_providers.run()

    updated = controller_with_providers.service.get_by_name("mcp-provider-1-updated")
    assert updated is not None
    assert updated.endpoint == "http://localhost:9999/sse"
    assert controller_with_providers.service.get_by_name("mcp-provider-1") is None


@patch("data_designer.cli.controllers.mcp_provider_controller.print_error")
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_handle_update_with_no_providers(
    mock_select: MagicMock,
    mock_print_error: MagicMock,
    controller: MCPProviderController,
) -> None:
    """Test update mode with no providers shows error."""
    mock_select.side_effect = ["update"]

    # First need to add a provider so we get past the initial check
    # Actually, we need to mock the existing providers check
    with patch.object(controller.service, "list_all", side_effect=[[], []]):
        controller._handle_update()

    mock_print_error.assert_called_once_with("No MCP providers to update")


@patch("data_designer.cli.controllers.mcp_provider_controller.print_error")
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_handle_delete_with_no_providers(
    mock_select: MagicMock,
    mock_print_error: MagicMock,
    controller: MCPProviderController,
) -> None:
    """Test delete mode with no providers shows error."""
    controller._handle_delete()

    mock_print_error.assert_called_once_with("No MCP providers to delete")


def test_mask_api_keys_masks_regular_keys(controller: MCPProviderController) -> None:
    """Test _mask_api_keys masks regular API keys but not env vars."""
    config = {
        "providers": [
            {"name": "test", "api_key": "secret12345678"},
            {"name": "test2", "api_key": "SOME_ENV_VAR"},
            {"name": "test3", "api_key": "tiny"},
            {"name": "test4", "api_key": None},
        ]
    }

    masked = controller._mask_api_keys(config)

    # Regular API key should be masked
    assert masked["providers"][0]["api_key"] == "***5678"
    # Uppercase env var names should stay visible
    assert masked["providers"][1]["api_key"] == "SOME_ENV_VAR"
    # Short keys should be fully masked
    assert masked["providers"][2]["api_key"] == "***"
    # None should stay None
    assert masked["providers"][3]["api_key"] is None


def test_select_provider_handles_sse_provider(
    controller_with_providers: MCPProviderController,
    stub_mcp_providers: list[MCPProviderT],
) -> None:
    """Test _select_provider displays SSE provider correctly."""
    sse_providers = [p for p in stub_mcp_providers if isinstance(p, MCPProvider)]

    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.select_with_arrows",
        return_value=sse_providers[0].name,
    ) as mock_select:
        result = controller_with_providers._select_provider(sse_providers, "Select provider")

    assert result == sse_providers[0].name
    # Check that the options were built correctly
    call_args = mock_select.call_args
    options = call_args[0][0]
    assert f"(SSE: {sse_providers[0].endpoint})" in options[sse_providers[0].name]


def test_select_provider_handles_stdio_provider(
    controller_with_providers: MCPProviderController,
    stub_mcp_providers: list[MCPProviderT],
) -> None:
    """Test _select_provider displays stdio provider correctly."""
    stdio_providers = [p for p in stub_mcp_providers if isinstance(p, LocalStdioMCPProvider)]

    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.select_with_arrows",
        return_value=stdio_providers[0].name,
    ) as mock_select:
        result = controller_with_providers._select_provider(stdio_providers, "Select provider")

    assert result == stdio_providers[0].name
    # Check that the options were built correctly
    call_args = mock_select.call_args
    options = call_args[0][0]
    assert f"(stdio: {stdio_providers[0].command})" in options[stdio_providers[0].name]


@patch("data_designer.cli.controllers.mcp_provider_controller.confirm_action", return_value=False)
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_delete_provider_cancelled_by_user(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test delete is cancelled when user declines confirmation."""
    mock_select.side_effect = ["delete", "mcp-provider-1"]
    initial_count = len(controller_with_providers.service.list_all())

    controller_with_providers.run()

    # Provider should not be deleted
    assert len(controller_with_providers.service.list_all()) == initial_count


@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows", return_value=None)
def test_select_mode_returns_none_when_cancelled(
    mock_select: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test _select_mode returns None when user cancels."""
    result = controller_with_providers._select_mode()
    assert result is None


def test_confirm_add_another_yes(controller: MCPProviderController) -> None:
    """Test _confirm_add_another returns True for yes."""
    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.select_with_arrows",
        return_value="yes",
    ):
        assert controller._confirm_add_another() is True


def test_confirm_add_another_no(controller: MCPProviderController) -> None:
    """Test _confirm_add_another returns False for no."""
    with patch(
        "data_designer.cli.controllers.mcp_provider_controller.select_with_arrows",
        return_value="no",
    ):
        assert controller._confirm_add_another() is False


@patch("data_designer.cli.controllers.mcp_provider_controller.print_error")
@patch("data_designer.cli.controllers.mcp_provider_controller.select_with_arrows")
def test_handle_update_provider_not_found(
    mock_select: MagicMock,
    mock_print_error: MagicMock,
    controller_with_providers: MCPProviderController,
) -> None:
    """Test update shows error when selected provider not found."""
    mock_select.side_effect = ["update", "nonexistent-provider"]

    controller_with_providers.run()

    mock_print_error.assert_called_with("MCP provider 'nonexistent-provider' not found")


@patch("data_designer.cli.controllers.mcp_provider_controller.print_error")
def test_handle_add_catches_value_error(
    mock_print_error: MagicMock,
    controller: MCPProviderController,
) -> None:
    """Test add handles ValueError from service."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = MCPProvider(name="test", endpoint="http://localhost:8080/sse")

    with (
        patch(
            "data_designer.cli.controllers.mcp_provider_controller.MCPProviderFormBuilder",
            return_value=mock_builder,
        ),
        patch.object(controller.service, "add", side_effect=ValueError("Duplicate name")),
    ):
        controller._handle_add()

    mock_print_error.assert_called_once_with("Failed to add MCP provider: Duplicate name")
