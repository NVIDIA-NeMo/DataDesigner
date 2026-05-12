# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import click
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
        version=None,
        yes=True,
        dry_run=True,
    )


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_install_command_delegates_version_to_controller(mock_ctrl_cls: MagicMock) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(
        app,
        ["plugin", "install", "github", "--version", "0.1.0", "--dry-run"],
    )

    assert result.exit_code == 0
    mock_ctrl.run_install.assert_called_once_with(
        "github",
        catalog_alias=None,
        refresh=False,
        manager="auto",
        version="0.1.0",
        yes=False,
        dry_run=True,
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
    output = click.unstyle(result.output)

    assert result.exit_code == 0
    assert "PACKAGE" in output
    assert "Plugin package name or package alias" in output
    assert "runtime plugin name" not in output


def test_plugin_install_help_uses_package_first_wording() -> None:
    result = runner.invoke(app, ["plugin", "install", "--help"])
    output = click.unstyle(result.output)

    assert result.exit_code == 0
    assert "PACKAGE" in output
    assert "Plugin package name or package alias" in output
    assert "--version" in output
    assert "runtime plugin name" not in output
    assert "Print the install plan" in output


def test_plugin_uninstall_help_uses_package_first_wording() -> None:
    result = runner.invoke(app, ["plugin", "uninstall", "--help"])
    output = click.unstyle(result.output)

    assert result.exit_code == 0
    assert "PACKAGE" in output
    assert "Plugin package name or package alias" in output
    assert "runtime plugin name" not in output
    assert "Print the uninstall plan" in output


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
        ],
    )

    assert result.exit_code == 0
    mock_ctrl.run_catalog_add.assert_called_once_with(
        alias="research",
        url="https://github.com/acme/dd-plugins",
    )


@patch("data_designer.cli.commands.plugin.print_warning")
@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_installed_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_warning: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "--catalog", "research", "installed"])

    assert result.exit_code == 0
    mock_print_warning.assert_called_once_with(
        "Ignoring --catalog 'research'; installed plugin packages are discovered from the current Python environment."
    )
    mock_ctrl.run_installed.assert_called_once_with()


@patch("data_designer.cli.commands.plugin.print_warning")
@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_plugin_catalog_list_warns_when_parent_catalog_is_unused(
    mock_ctrl_cls: MagicMock,
    mock_print_warning: MagicMock,
) -> None:
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    result = runner.invoke(app, ["plugin", "--catalog", "research", "catalog", "list"])

    assert result.exit_code == 0
    mock_print_warning.assert_called_once_with(
        "Ignoring --catalog 'research'; catalog management commands operate on aliases directly."
    )
    mock_ctrl.run_catalog_list.assert_called_once_with()
