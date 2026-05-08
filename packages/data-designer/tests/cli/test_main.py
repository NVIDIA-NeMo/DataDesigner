# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from unittest.mock import Mock, call, patch

from typer.testing import CliRunner

from data_designer.cli.main import app, main
from data_designer.cli.version_notice import UpdateNotice
from data_designer.config.utils.constants import DEFAULT_NUM_RECORDS
from data_designer.engine.storage.artifact_storage import ResumeMode

runner = CliRunner()


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_bootstraps_before_running_app(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint bootstraps defaults before invoking Typer."""
    call_order = Mock()
    call_order.attach_mock(mock_bootstrap, "bootstrap")
    call_order.attach_mock(mock_app, "app")

    with patch("sys.argv", ["data-designer"]):
        main()

    assert call_order.mock_calls == [call.bootstrap(), call.app()]


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_bootstraps_for_plugins_commands(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """Plugin commands still run through CLI default setup before Typer dispatch."""
    with patch("sys.argv", ["data-designer", "plugins", "list"]):
        main()

    mock_bootstrap.assert_called_once_with()
    mock_app.assert_called_once_with()


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_skips_bootstrap_for_version(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint avoids default setup for the fast version path."""
    with patch("sys.argv", ["data-designer", "--version"]):
        main()

    mock_bootstrap.assert_not_called()
    mock_app.assert_called_once_with()


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_skips_bootstrap_when_version_follows_another_flag(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The CLI entrypoint detects eager version requests even after another root flag."""
    with patch("sys.argv", ["data-designer", "--help", "--version"]):
        main()

    mock_bootstrap.assert_not_called()
    mock_app.assert_called_once_with()


def test_app_version_prints_installed_data_designer_version() -> None:
    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0") as mock_version,
        patch("data_designer.cli.main.should_show_update_notice", return_value=True),
        patch("data_designer.cli.version_notice.get_update_notice", return_value=None) as mock_notice,
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"
    mock_version.assert_called_once_with("data-designer")
    mock_notice.assert_called_once_with("0.6.0")


def test_app_version_prints_update_notice_after_installed_version() -> None:
    notice = UpdateNotice(latest_version="0.6.1", upgrade_command="uv tool upgrade data-designer")
    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0"),
        patch("data_designer.cli.main.should_show_update_notice", return_value=True),
        patch("data_designer.cli.version_notice.get_update_notice", return_value=notice),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output.splitlines()[0] == "0.6.0"
    assert "New Data Designer version available: 0.6.1" in result.output
    assert "Upgrade with: uv tool upgrade data-designer" in result.output


def test_app_version_skips_update_notice_when_stdout_is_not_tty() -> None:
    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0"),
        patch("data_designer.cli.main.should_show_update_notice", return_value=False),
        patch("data_designer.cli.version_notice.get_update_notice") as mock_notice,
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"
    mock_notice.assert_not_called()


def test_app_version_skips_update_notice_when_lookup_fails() -> None:
    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0"),
        patch("data_designer.cli.main.should_show_update_notice", return_value=True),
        patch("data_designer.cli.version_notice.get_update_notice", side_effect=RuntimeError("network failure")),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"


def test_app_version_skips_update_notice_when_lazy_import_fails() -> None:
    real_import = __import__

    def fail_ui_import(name: str, *args: object, **kwargs: object) -> object:
        if name == "data_designer.cli.ui":
            raise ImportError("ui unavailable")
        return real_import(name, *args, **kwargs)

    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0"),
        patch("data_designer.cli.main.should_show_update_notice", return_value=True),
        patch("builtins.__import__", side_effect=fail_ui_import),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"


def test_app_version_skips_update_notice_when_render_fails() -> None:
    notice = UpdateNotice(latest_version="0.6.1", upgrade_command="uv tool upgrade data-designer")
    with (
        patch("data_designer.cli.main.importlib.metadata.version", return_value="0.6.0"),
        patch("data_designer.cli.main.should_show_update_notice", return_value=True),
        patch("data_designer.cli.version_notice.get_update_notice", return_value=notice),
        patch("data_designer.cli.ui.print_update_notice", side_effect=RuntimeError("render failure")),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert result.output == "0.6.0\n"


def test_app_version_errors_when_package_version_is_missing() -> None:
    with patch(
        "data_designer.cli.main.importlib.metadata.version",
        side_effect=importlib.metadata.PackageNotFoundError("data-designer"),
    ):
        result = runner.invoke(app, ["--version"])

    assert result.exit_code == 1
    assert "Unable to resolve installed data-designer package version." in result.output


@patch("data_designer.cli.commands.create.GenerationController")
def test_app_dispatches_lazy_create_command(mock_controller_cls: Mock) -> None:
    """The Typer app dispatches lazy-loaded commands through the resolved callback."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["create", "config.yaml"])

    assert result.exit_code == 0
    mock_controller.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=DEFAULT_NUM_RECORDS,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
    )


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_app_dispatches_lazy_plugins_list_command(mock_controller_cls: Mock) -> None:
    """The plugins group lazily resolves command callbacks without loading a catalog."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(
        app,
        ["plugins", "--catalog", "local", "list", "--refresh", "--include-incompatible"],
    )

    assert result.exit_code == 0
    mock_controller.run_list.assert_called_once_with(
        catalog_alias="local",
        refresh=True,
        include_incompatible=True,
    )


@patch("data_designer.cli.commands.plugins.PluginCatalogController")
def test_app_dispatches_lazy_plugin_catalogs_list_command(mock_controller_cls: Mock) -> None:
    """Nested plugin catalog commands resolve through the lazy command group."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["plugins", "catalogs", "list"])

    assert result.exit_code == 0
    mock_controller.run_catalogs_list.assert_called_once_with()


def test_app_help_keeps_config_and_plugins_commands_reachable() -> None:
    config_result = runner.invoke(app, ["config", "--help"])
    plugins_result = runner.invoke(app, ["plugins", "--help"])

    assert config_result.exit_code == 0
    assert "providers" in config_result.output
    assert "models" in config_result.output
    assert plugins_result.exit_code == 0
    assert "list" in plugins_result.output
    assert "install" in plugins_result.output
    assert "uninstall" in plugins_result.output
    assert "catalogs" in plugins_result.output
