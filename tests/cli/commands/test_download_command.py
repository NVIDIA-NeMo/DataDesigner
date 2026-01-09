# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.download import personas_command
from data_designer.cli.controllers.download_controller import DownloadController
from data_designer.config.utils.constants import DATA_DESIGNER_HOME


@patch("data_designer.cli.commands.download.DownloadController")
def test_personas_command_interactive_mode(mock_download_controller: MagicMock) -> None:
    """Test personas_command with no arguments (interactive mode)."""
    mock_controller_instance = MagicMock(spec=DownloadController)
    mock_download_controller.return_value = mock_controller_instance

    personas_command(locales=None, all_locales=False, dry_run=False, list_available=False)

    mock_download_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller_instance.run_personas.assert_called_once_with(locales=None, all_locales=False, dry_run=False)


@patch("data_designer.cli.commands.download.DownloadController")
def test_personas_command_with_specific_locales(mock_download_controller: MagicMock) -> None:
    """Test personas_command with --locale flags."""
    mock_controller_instance = MagicMock(spec=DownloadController)
    mock_download_controller.return_value = mock_controller_instance

    personas_command(locales=["en_US", "ja_JP"], all_locales=False, dry_run=False, list_available=False)

    mock_download_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller_instance.run_personas.assert_called_once_with(
        locales=["en_US", "ja_JP"], all_locales=False, dry_run=False
    )


@patch("data_designer.cli.commands.download.DownloadController")
def test_personas_command_with_all_flag(mock_download_controller: MagicMock) -> None:
    """Test personas_command with --all flag."""
    mock_controller_instance = MagicMock(spec=DownloadController)
    mock_download_controller.return_value = mock_controller_instance

    personas_command(locales=None, all_locales=True, dry_run=False, list_available=False)

    mock_download_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller_instance.run_personas.assert_called_once_with(locales=None, all_locales=True, dry_run=False)


@patch("data_designer.cli.commands.download.DownloadController")
def test_personas_command_with_dry_run_flag(mock_download_controller: MagicMock) -> None:
    """Test personas_command with --dry-run flag."""
    mock_controller_instance = MagicMock(spec=DownloadController)
    mock_download_controller.return_value = mock_controller_instance

    personas_command(locales=["en_US"], all_locales=False, dry_run=True, list_available=False)

    mock_download_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller_instance.run_personas.assert_called_once_with(locales=["en_US"], all_locales=False, dry_run=True)


@patch("data_designer.cli.commands.download.DownloadController")
def test_personas_command_with_list_flag(mock_download_controller: MagicMock) -> None:
    """Test personas_command with --list flag."""
    mock_controller_instance = MagicMock(spec=DownloadController)
    mock_download_controller.return_value = mock_controller_instance

    personas_command(locales=None, all_locales=False, dry_run=False, list_available=True)

    mock_download_controller.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_controller_instance.list_personas.assert_called_once()
    mock_controller_instance.run_personas.assert_not_called()
