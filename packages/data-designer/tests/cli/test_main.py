# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from data_designer.cli.main import app

runner = CliRunner()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
@patch("data_designer.cli.commands.create.GenerationController")
def test_cli_bootstraps_for_create_command(mock_controller_cls: MagicMock, mock_bootstrap: MagicMock) -> None:
    """CLI bootstrap runs before create command execution."""
    mock_controller = MagicMock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["create", "config.yaml"])

    assert result.exit_code == 0
    mock_bootstrap.assert_called_once_with()
    mock_controller.run_create.assert_called_once()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
@patch("data_designer.cli.commands.preview.GenerationController")
def test_cli_bootstraps_for_preview_command(mock_controller_cls: MagicMock, mock_bootstrap: MagicMock) -> None:
    """CLI bootstrap runs before preview command execution."""
    mock_controller = MagicMock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["preview", "config.yaml", "--non-interactive"])

    assert result.exit_code == 0
    mock_bootstrap.assert_called_once_with()
    mock_controller.run_preview.assert_called_once()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
@patch("data_designer.cli.commands.models.ModelController")
def test_cli_bootstraps_for_config_models_command(mock_controller_cls: MagicMock, mock_bootstrap: MagicMock) -> None:
    """CLI bootstrap runs before config models command execution."""
    mock_controller = MagicMock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["config", "models"])

    assert result.exit_code == 0
    mock_bootstrap.assert_called_once_with()
    mock_controller.run.assert_called_once()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
@patch("data_designer.cli.commands.download.DownloadController")
def test_cli_bootstraps_for_download_command(mock_controller_cls: MagicMock, mock_bootstrap: MagicMock) -> None:
    """CLI bootstrap runs before download personas command execution."""
    mock_controller = MagicMock()
    mock_controller_cls.return_value = mock_controller

    result = runner.invoke(app, ["download", "personas", "--list"])

    assert result.exit_code == 0
    mock_bootstrap.assert_called_once_with()
    mock_controller.list_personas.assert_called_once()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
def test_cli_help_does_not_bootstrap(mock_bootstrap: MagicMock) -> None:
    """Top-level help remains side-effect free."""
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    mock_bootstrap.assert_not_called()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
def test_config_help_does_not_bootstrap(mock_bootstrap: MagicMock) -> None:
    """Config group help remains side-effect free."""
    result = runner.invoke(app, ["config", "--help"])

    assert result.exit_code == 0
    mock_bootstrap.assert_not_called()


@patch("data_designer.cli.lazy_group.ensure_cli_default_model_settings")
def test_download_help_does_not_bootstrap(mock_bootstrap: MagicMock) -> None:
    """Download group help remains side-effect free."""
    result = runner.invoke(app, ["download", "--help"])

    assert result.exit_code == 0
    mock_bootstrap.assert_not_called()
