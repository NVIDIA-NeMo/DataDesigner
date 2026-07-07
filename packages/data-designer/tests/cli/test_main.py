# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.metadata
from unittest.mock import Mock, call, patch

import click
from typer.testing import CliRunner

from data_designer.cli.main import app, main
from data_designer.cli.version_notice import UpdateNotice
from data_designer.config.utils.constants import DEFAULT_DISPLAY_WIDTH, DEFAULT_NUM_RECORDS
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
def test_main_bootstraps_for_plugin_commands(mock_bootstrap: Mock, mock_app: Mock) -> None:
    """The plugin command still runs through CLI default setup before Typer dispatch."""
    with patch("sys.argv", ["data-designer", "plugin", "list"]):
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


@patch("data_designer.cli.main.app")
@patch("data_designer.cli.main.ensure_cli_default_model_settings")
def test_main_bootstraps_when_version_is_forwarded(mock_bootstrap: Mock, mock_app: Mock) -> None:
    with patch("sys.argv", ["data-designer", "create", "workflow.py", "--", "--version"]):
        main()

    mock_bootstrap.assert_called_once_with()
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
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_app_dispatches_create_tui_flags(mock_controller_cls: Mock) -> None:
    """The create command parses --tui/--no-tui into the TUI override."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["create", "config.yaml", "--no-tui"])

    assert result.exit_code == 0
    mock_controller.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=DEFAULT_NUM_RECORDS,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=False,
        script_args=None,
    )

    mock_controller.reset_mock()

    result = runner.invoke(app, ["create", "config.yaml", "--tui"])

    assert result.exit_code == 0
    mock_controller.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=DEFAULT_NUM_RECORDS,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=True,
        script_args=None,
    )


@patch("data_designer.cli.commands.validate.GenerationController")
@patch("data_designer.cli.commands.preview.GenerationController")
@patch("data_designer.cli.commands.create.GenerationController")
def test_app_dispatches_script_args_for_generation_commands(
    mock_create_cls: Mock,
    mock_preview_cls: Mock,
    mock_validate_cls: Mock,
) -> None:
    create_result = runner.invoke(
        app,
        [
            "create",
            "workflow.py",
            "--num-records",
            "32",
            "--dataset-name",
            "run_a",
            "--",
            "--seed-path",
            "seeds.parquet",
        ],
    )
    preview_result = runner.invoke(
        app,
        ["preview", "workflow.py", "-n", "3", "--", "--seed-path", "sample.parquet"],
    )
    validate_result = runner.invoke(
        app,
        ["validate", "workflow.py", "--", "--seed-path", "seeds.parquet"],
    )

    assert create_result.exit_code == preview_result.exit_code == validate_result.exit_code == 0
    mock_create_cls.return_value.run_create.assert_called_once_with(
        config_source="workflow.py",
        num_records=32,
        dataset_name="run_a",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=["--seed-path", "seeds.parquet"],
    )
    mock_preview_cls.return_value.run_preview.assert_called_once_with(
        config_source="workflow.py",
        num_records=3,
        non_interactive=False,
        save_results=False,
        artifact_path=None,
        theme="dark",
        display_width=DEFAULT_DISPLAY_WIDTH,
        script_args=["--seed-path", "sample.parquet"],
    )
    mock_validate_cls.return_value.run_validate.assert_called_once_with(
        config_source="workflow.py",
        script_args=["--seed-path", "seeds.parquet"],
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_app_rejects_unknown_options_before_script_separator(mock_controller_cls: Mock) -> None:
    result = runner.invoke(
        app,
        ["create", "workflow.py", "--num-recrods", "32", "--", "--custom", "value"],
    )

    assert result.exit_code == 2
    assert "No such option: --num-recrods" in click.unstyle(result.output)
    mock_controller_cls.assert_not_called()


@patch("data_designer.cli.commands.check_models.GenerationController")
def test_app_dispatches_lazy_check_models_command(mock_controller_cls: Mock) -> None:
    """The Typer app dispatches the lazy-loaded check-models command."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["check-models", "config.yaml"])

    assert result.exit_code == 0
    mock_controller.run_check_models.assert_called_once_with(config_source="config.yaml")


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_app_dispatches_lazy_plugin_list_command(mock_controller_cls: Mock) -> None:
    """The plugin group lazily resolves command callbacks without loading a catalog."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(
        app,
        ["plugin", "--catalog", "local", "list", "--refresh", "--include-incompatible"],
    )

    assert result.exit_code == 0
    mock_controller.run_list.assert_called_once_with(
        catalog_alias="local",
        refresh=True,
        include_incompatible=True,
    )


@patch("data_designer.cli.commands.plugin.PluginCatalogController")
def test_app_dispatches_lazy_plugin_catalog_list_command(mock_controller_cls: Mock) -> None:
    """Nested plugin catalog commands resolve through the lazy command group."""
    mock_controller = Mock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["plugin", "catalog", "list"])

    assert result.exit_code == 0
    mock_controller.run_catalog_list.assert_called_once_with()


def test_app_help_keeps_config_and_plugin_commands_reachable() -> None:
    config_result = runner.invoke(app, ["config", "--help"])
    plugin_result = runner.invoke(app, ["plugin", "--help"])

    assert config_result.exit_code == 0
    assert "providers" in config_result.output
    assert "models" in config_result.output
    assert plugin_result.exit_code == 0
    assert "list" in plugin_result.output
    assert "install" in plugin_result.output
    assert "uninstall" in plugin_result.output
    assert "catalog" in plugin_result.output
    assert "install strategy" in plugin_result.output
    assert "installed plugin packages" in plugin_result.output
    assert "runtime plugins and their packages" not in plugin_result.output


def test_no_args_help_exits_successfully_for_lazy_groups() -> None:
    root_result = runner.invoke(app, [])
    plugin_result = runner.invoke(app, ["plugin"])
    plugin_catalog_result = runner.invoke(app, ["plugin", "catalog"])

    assert root_result.exit_code == 0
    assert "Data Designer CLI" in root_result.output
    assert "plugin" in root_result.output
    assert plugin_result.exit_code == 0
    assert "Discover, install, and uninstall" in plugin_result.output
    assert "catalog" in plugin_result.output
    assert plugin_catalog_result.exit_code == 0
    assert "Manage plugin catalog aliases" in plugin_catalog_result.output
    assert "add" in plugin_catalog_result.output


def test_app_does_not_expose_legacy_plugins_command() -> None:
    result = runner.invoke(app, ["plugins", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output


def test_plugin_does_not_expose_legacy_catalogs_command() -> None:
    result = runner.invoke(app, ["plugin", "catalogs", "--help"])

    assert result.exit_code != 0
    assert "No such command" in result.output
