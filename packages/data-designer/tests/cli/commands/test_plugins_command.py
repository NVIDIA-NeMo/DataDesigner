# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_list_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugins", "--catalog", "research", "list", "--refresh", "--include-incompatible"])

    assert result.exit_code == 0
    mock_ctrl.run_list.assert_called_once_with(
        catalog_alias="research",
        refresh=True,
        include_incompatible=True,
    )


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_search_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugins", "search", "github", "--catalog", "research"])

    assert result.exit_code == 0
    mock_ctrl.run_search.assert_called_once_with(
        "github",
        catalog_alias="research",
        refresh=False,
        include_incompatible=False,
    )


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_install_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugins", "install", "text-transform", "--manager", "pip", "--yes", "--dry-run"])

    assert result.exit_code == 0
    mock_ctrl.run_install.assert_called_once_with(
        "text-transform",
        catalog_alias=None,
        refresh=False,
        manager="pip",
        yes=True,
        dry_run=True,
        force=False,
    )


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_catalogs_add_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(
        app,
        [
            "plugins",
            "catalogs",
            "add",
            "research",
            "https://github.com/acme/dd-plugins",
            "--trusted",
            "--cache-ttl-seconds",
            "60",
        ],
    )

    assert result.exit_code == 0
    mock_ctrl.run_catalogs_add.assert_called_once_with(
        alias="research",
        url="https://github.com/acme/dd-plugins",
        trusted=True,
        cache_ttl_seconds=60,
    )


@patch("data_designer.cli.commands.plugins.print_info")
@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_installed_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_info: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugins", "--catalog", "research", "installed"])

    assert result.exit_code == 0
    mock_print_info.assert_called_once_with(
        "Ignoring --catalog 'research'; installed plugins are discovered from the current Python environment."
    )
    mock_ctrl.run_installed.assert_called_once_with()


@patch("data_designer.cli.commands.plugins.print_info")
@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_plugins_catalogs_list_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_info: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugins", "--catalog", "research", "catalogs", "list"])

    assert result.exit_code == 0
    mock_print_info.assert_called_once_with(
        "Ignoring --catalog 'research'; catalog management commands operate on aliases directly."
    )
    mock_ctrl.run_catalogs_list.assert_called_once_with()
