# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.download_controller import DownloadController


@pytest.fixture
def controller(tmp_path: Path) -> DownloadController:
    """Create a controller instance for testing."""
    return DownloadController(tmp_path)


@pytest.fixture
def controller_with_datasets(tmp_path: Path) -> DownloadController:
    """Create a controller instance with existing datasets."""
    controller = DownloadController(tmp_path)
    # Create managed assets directory with sample parquet files
    managed_assets_dir = tmp_path / "managed-assets" / "datasets"
    managed_assets_dir.mkdir(parents=True, exist_ok=True)
    (managed_assets_dir / "en_US.parquet").touch()
    return controller


def test_init(tmp_path: Path) -> None:
    """Test controller initialization sets up service correctly."""
    controller = DownloadController(tmp_path)
    assert controller.config_dir == tmp_path
    assert controller.service.config_dir == tmp_path
    assert controller.persona_repository is not None
    assert controller.service.persona_repository is controller.persona_repository


def test_list_personas(controller: DownloadController) -> None:
    """Test list_personas displays all available datasets."""
    controller.list_personas()
    # Method should complete without errors and print to console


def test_list_personas_with_downloaded_datasets(controller_with_datasets: DownloadController) -> None:
    """Test list_personas shows (downloaded) status for existing datasets."""
    controller_with_datasets.list_personas()
    # Method should complete without errors and show downloaded status


@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=False)
@patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows", return_value=["en_US"])
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_user_cancels_confirmation(
    mock_check_ngc: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas when user cancels at confirmation prompt."""
    controller.run_personas(locales=None, all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Verify interactive selection was called
    mock_select.assert_called_once()

    # Verify confirmation was requested
    mock_confirm.assert_called_once()


@patch.object(DownloadController, "_download_locale", return_value=True)
@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_with_all_flag(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with --all flag downloads all locales."""
    controller.run_personas(locales=None, all_locales=True)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Verify all 7 locales were downloaded
    assert mock_download.call_count == 7

    # Verify each locale was downloaded
    downloaded_locales = [call[0][0] for call in mock_download.call_args_list]
    assert "en_US" in downloaded_locales
    assert "en_IN" in downloaded_locales
    assert "en_SG" in downloaded_locales
    assert "hi_Deva_IN" in downloaded_locales
    assert "hi_Latn_IN" in downloaded_locales
    assert "ja_JP" in downloaded_locales
    assert "pt_BR" in downloaded_locales


@patch.object(DownloadController, "_download_locale", return_value=True)
@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_with_specific_locales(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with specific locale flags."""
    controller.run_personas(locales=["en_US", "ja_JP"], all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Verify only specified locales were downloaded
    assert mock_download.call_count == 2
    downloaded_locales = [call[0][0] for call in mock_download.call_args_list]
    assert "en_US" in downloaded_locales
    assert "ja_JP" in downloaded_locales


@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_with_invalid_locales(
    mock_check_ngc: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with invalid locale codes."""
    controller.run_personas(locales=["invalid_locale", "en_US"], all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Function should exit early without attempting download


@patch.object(DownloadController, "_download_locale", return_value=True)
@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows", return_value=["en_US"])
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_interactive_selection(
    mock_check_ngc: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with interactive locale selection."""
    controller.run_personas(locales=None, all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Verify interactive selection was called
    mock_select.assert_called_once()

    # Verify confirmation was requested
    mock_confirm.assert_called_once()

    # Verify selected locale was downloaded
    mock_download.assert_called_once_with("en_US")


@patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows", return_value=None)
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_interactive_cancelled(
    mock_check_ngc: MagicMock,
    mock_select: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas when user cancels interactive selection."""
    controller.run_personas(locales=None, all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()

    # Verify interactive selection was called
    mock_select.assert_called_once()

    # Function should exit early


@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=False)
def test_run_personas_ngc_cli_not_available(
    mock_check_ngc: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas exits early when NGC CLI is not available."""
    controller.run_personas(locales=["en_US"], all_locales=False)

    # Verify NGC check was called
    mock_check_ngc.assert_called_once()


def test_check_ngc_cli_available_with_version() -> None:
    """Test check_ngc_cli_with_instructions displays version when NGC CLI is available."""
    from data_designer.cli.controllers.download_controller import check_ngc_cli_with_instructions

    with patch("data_designer.cli.controllers.download_controller.check_ngc_cli_available", return_value=True):
        with patch("data_designer.cli.controllers.download_controller.get_ngc_version", return_value="NGC CLI 3.41.4"):
            result = check_ngc_cli_with_instructions()

    assert result is True


def test_check_ngc_cli_available_without_version() -> None:
    """Test check_ngc_cli_with_instructions when version cannot be determined."""
    from data_designer.cli.controllers.download_controller import check_ngc_cli_with_instructions

    with patch("data_designer.cli.controllers.download_controller.check_ngc_cli_available", return_value=True):
        with patch("data_designer.cli.controllers.download_controller.get_ngc_version", return_value=None):
            result = check_ngc_cli_with_instructions()

    assert result is True


def test_determine_locales_with_all_flag(controller: DownloadController) -> None:
    """Test _determine_locales returns all locales when all_locales=True."""
    result = controller._determine_locales(locales=None, all_locales=True)

    assert len(result) == 7
    assert "en_US" in result
    assert "en_IN" in result
    assert "en_SG" in result
    assert "hi_Deva_IN" in result
    assert "hi_Latn_IN" in result
    assert "ja_JP" in result
    assert "pt_BR" in result


def test_determine_locales_with_valid_locale_flags(controller: DownloadController) -> None:
    """Test _determine_locales with valid locale flags."""
    result = controller._determine_locales(locales=["en_US", "ja_JP"], all_locales=False)

    assert result == ["en_US", "ja_JP"]


def test_determine_locales_with_invalid_locale_flags(controller: DownloadController) -> None:
    """Test _determine_locales with invalid locale flags."""
    result = controller._determine_locales(locales=["invalid", "en_US"], all_locales=False)

    assert result == []


@patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows", return_value=["en_US"])
def test_determine_locales_interactive(mock_select: MagicMock, controller: DownloadController) -> None:
    """Test _determine_locales with interactive selection."""
    result = controller._determine_locales(locales=None, all_locales=False)

    assert result == ["en_US"]
    mock_select.assert_called_once()


def test_select_locales_interactive(controller: DownloadController) -> None:
    """Test _select_locales_interactive calls UI function correctly."""
    available_locales = controller.service.get_available_locales()

    with patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows") as mock_select:
        mock_select.return_value = ["en_US", "ja_JP"]
        result = controller._select_locales_interactive(available_locales)

    assert result == ["en_US", "ja_JP"]
    mock_select.assert_called_once()


def test_download_locale_success(controller: DownloadController) -> None:
    """Test _download_locale successfully downloads a locale."""
    with patch.object(controller.service, "download_persona_dataset"):
        result = controller._download_locale("en_US")

    assert result is True


def test_download_locale_subprocess_error(controller: DownloadController) -> None:
    """Test _download_locale handles subprocess errors."""
    with patch.object(
        controller.service,
        "download_persona_dataset",
        side_effect=subprocess.CalledProcessError(1, "ngc"),
    ):
        result = controller._download_locale("en_US")

    assert result is False


def test_download_locale_generic_error(controller: DownloadController) -> None:
    """Test _download_locale handles generic errors."""
    with patch.object(
        controller.service,
        "download_persona_dataset",
        side_effect=Exception("Unexpected error"),
    ):
        result = controller._download_locale("en_US")

    assert result is False


@patch.object(DownloadController, "_download_locale")
@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_mixed_success_and_failure(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with mixed success and failure results."""
    # First download succeeds, second fails
    mock_download.side_effect = [True, False]

    controller.run_personas(locales=["en_US", "ja_JP"], all_locales=False)

    # Verify both locales were attempted
    assert mock_download.call_count == 2


@patch.object(DownloadController, "_download_locale", return_value=True)
@patch("data_designer.cli.controllers.download_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions", return_value=True)
def test_run_personas_shows_existing_status(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller_with_datasets: DownloadController,
) -> None:
    """Test run_personas shows (already exists, will update) status for existing datasets."""
    controller_with_datasets.run_personas(locales=["en_US"], all_locales=False)

    # Verify download was attempted (it would show the "already exists" message)
    mock_download.assert_called_once_with("en_US")


@patch.object(DownloadController, "_download_locale")
@patch("data_designer.cli.controllers.download_controller.confirm_action")
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions")
def test_run_personas_with_dry_run_flag(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with --dry-run flag does not download or check NGC CLI."""
    controller.run_personas(locales=["en_US", "ja_JP"], all_locales=False, dry_run=True)

    # Verify NGC check was NOT called in dry run mode
    mock_check_ngc.assert_not_called()

    # Verify confirmation was NOT requested in dry run mode
    mock_confirm.assert_not_called()

    # Verify no downloads were attempted
    mock_download.assert_not_called()


@patch.object(DownloadController, "_download_locale")
@patch("data_designer.cli.controllers.download_controller.confirm_action")
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions")
def test_run_personas_with_all_and_dry_run(
    mock_check_ngc: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with --all and --dry-run flags."""
    controller.run_personas(locales=None, all_locales=True, dry_run=True)

    # Verify NGC check was NOT called in dry run mode
    mock_check_ngc.assert_not_called()

    # Verify confirmation was NOT requested in dry run mode
    mock_confirm.assert_not_called()

    # Verify no downloads were attempted
    mock_download.assert_not_called()


@patch.object(DownloadController, "_download_locale")
@patch("data_designer.cli.controllers.download_controller.confirm_action")
@patch("data_designer.cli.controllers.download_controller.select_multiple_with_arrows", return_value=["en_US"])
@patch("data_designer.cli.controllers.download_controller.check_ngc_cli_with_instructions")
def test_run_personas_interactive_with_dry_run(
    mock_check_ngc: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    mock_download: MagicMock,
    controller: DownloadController,
) -> None:
    """Test run_personas with interactive selection and --dry-run flag."""
    controller.run_personas(locales=None, all_locales=False, dry_run=True)

    # Verify NGC check was NOT called in dry run mode
    mock_check_ngc.assert_not_called()

    # Verify interactive selection WAS called (user still needs to select)
    mock_select.assert_called_once()

    # Verify confirmation was NOT requested in dry run mode
    mock_confirm.assert_not_called()

    # Verify no downloads were attempted
    mock_download.assert_not_called()
