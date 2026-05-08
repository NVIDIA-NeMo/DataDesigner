# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_list_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "--catalog", "research", "list", "--refresh", "--include-incompatible"])

    assert result.exit_code == 0
    mock_ctrl.run_list.assert_called_once_with(
        catalog_alias="research",
        refresh=True,
        include_incompatible=True,
    )


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_search_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "search", "github", "--catalog", "research"])

    assert result.exit_code == 0
    mock_ctrl.run_search.assert_called_once_with(
        "github",
        catalog_alias="research",
        refresh=False,
        include_incompatible=False,
    )


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_install_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(
        app,
        ["plugin", "install", "data-designer-text-transform", "--manager", "pip", "--yes", "--dry-run"],
    )

    assert result.exit_code == 0
    mock_ctrl.run_install.assert_called_once_with(
        "data-designer-text-transform",
        catalog_alias=None,
        refresh=False,
        manager="pip",
        yes=True,
        dry_run=True,
        force=False,
    )


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_uninstall_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(
        app,
        ["plugin", "uninstall", "data-designer-text-transform", "--manager", "pip", "--yes", "--dry-run"],
    )

    assert result.exit_code == 0
    mock_ctrl.run_uninstall.assert_called_once_with(
        "data-designer-text-transform",
        catalog_alias=None,
        refresh=False,
        manager="pip",
        yes=True,
        dry_run=True,
    )


def test_plugin_info_help_uses_package_argument() -> None:
    result = runner.invoke(app, ["plugin", "info", "--help"])

    assert result.exit_code == 0
    assert "PACKAGE" in result.output
    assert "Plugin package name or package alias" in result.output
    assert "runtime plugin name" not in result.output


def test_plugin_install_help_uses_package_first_wording() -> None:
    result = runner.invoke(app, ["plugin", "install", "--help"])

    assert result.exit_code == 0
    assert "PACKAGE" in result.output
    assert "Plugin package name or package alias" in result.output
    assert "runtime plugin name" not in result.output
    assert "Allow installing a catalog" in result.output
    assert "package when compatibility" in result.output


def test_plugin_uninstall_help_uses_package_first_wording() -> None:
    result = runner.invoke(app, ["plugin", "uninstall", "--help"])

    assert result.exit_code == 0
    assert "PACKAGE" in result.output
    assert "Plugin package name or package alias" in result.output
    assert "runtime plugin name" not in result.output
    assert "Print the uninstall plan" in result.output


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_catalog_add_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(
        app,
        [
            "plugin",
            "catalog",
            "add",
            "research",
            "https://github.com/acme/dd-plugins",
            "--trusted",
            "--cache-ttl-seconds",
            "60",
        ],
    )

    assert result.exit_code == 0
    mock_ctrl.run_catalog_add.assert_called_once_with(
        alias="research",
        url="https://github.com/acme/dd-plugins",
        trusted=True,
        cache_ttl_seconds=60,
    )


@patch("data_designer.cli.commands.plugin.print_info")
@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_installed_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_info: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "--catalog", "research", "installed"])

    assert result.exit_code == 0
    mock_print_info.assert_called_once_with(
        "Ignoring --catalog 'research'; installed runtime plugins are discovered from the current Python environment."
    )
    mock_ctrl.run_installed.assert_called_once_with()


@patch("data_designer.cli.commands.plugin.print_info")
@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_catalog_list_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_info: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "--catalog", "research", "catalog", "list"])

    assert result.exit_code == 0
    mock_print_info.assert_called_once_with(
        "Ignoring --catalog 'research'; catalog management commands operate on aliases directly."
    )
    mock_ctrl.run_catalog_list.assert_called_once_with()
