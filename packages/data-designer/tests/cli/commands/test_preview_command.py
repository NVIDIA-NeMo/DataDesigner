# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.preview import preview_command
from data_designer.cli.ui import wait_for_navigation_key

# ---------------------------------------------------------------------------
# preview_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.commands.preview.GenerationController")
def test_preview_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """Test preview_command delegates to GenerationController.run_preview."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    preview_command(
        config_source="config.yaml", num_records=5, non_interactive=True, save_results=False, artifact_path=None
    )

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_preview.assert_called_once_with(
        config_source="config.yaml",
        num_records=5,
        non_interactive=True,
        save_results=False,
        artifact_path=None,
    )


@patch("data_designer.cli.commands.preview.GenerationController")
def test_preview_command_passes_non_interactive_false(mock_ctrl_cls: MagicMock) -> None:
    """Test preview_command passes non_interactive=False by default."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    preview_command(
        config_source="config.yaml", num_records=10, non_interactive=False, save_results=False, artifact_path=None
    )

    mock_ctrl.run_preview.assert_called_once_with(
        config_source="config.yaml",
        num_records=10,
        non_interactive=False,
        save_results=False,
        artifact_path=None,
    )


@patch("data_designer.cli.commands.preview.GenerationController")
def test_preview_command_passes_custom_num_records(mock_ctrl_cls: MagicMock) -> None:
    """Test preview_command passes custom num_records to controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    preview_command(
        config_source="my_config.py", num_records=20, non_interactive=True, save_results=False, artifact_path=None
    )

    mock_ctrl.run_preview.assert_called_once_with(
        config_source="my_config.py",
        num_records=20,
        non_interactive=True,
        save_results=False,
        artifact_path=None,
    )


@patch("data_designer.cli.commands.preview.GenerationController")
def test_preview_command_passes_save_results_and_artifact_path(mock_ctrl_cls: MagicMock) -> None:
    """Test preview_command passes save_results and artifact_path to controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    preview_command(
        config_source="config.yaml",
        num_records=5,
        non_interactive=True,
        save_results=True,
        artifact_path="/custom/output",
    )

    mock_ctrl.run_preview.assert_called_once_with(
        config_source="config.yaml",
        num_records=5,
        non_interactive=True,
        save_results=True,
        artifact_path="/custom/output",
    )


# ---------------------------------------------------------------------------
# wait_for_navigation_key unit tests (UI-layer, unchanged by refactor)
# ---------------------------------------------------------------------------


@patch("data_designer.cli.ui.Application")
def test_wait_for_navigation_key_creates_app_and_runs(mock_app_cls: MagicMock) -> None:
    """Test wait_for_navigation_key creates an Application and calls run()."""
    mock_app_instance = MagicMock()
    mock_app_cls.return_value = mock_app_instance

    result = wait_for_navigation_key()

    mock_app_cls.assert_called_once()
    mock_app_instance.run.assert_called_once()
    assert result == "q"


@patch("data_designer.cli.ui.Application")
def test_wait_for_navigation_key_handles_keyboard_interrupt(mock_app_cls: MagicMock) -> None:
    """Test wait_for_navigation_key returns 'q' on KeyboardInterrupt."""
    mock_app_instance = MagicMock()
    mock_app_cls.return_value = mock_app_instance
    mock_app_instance.run.side_effect = KeyboardInterrupt

    result = wait_for_navigation_key()

    assert result == "q"


@patch("data_designer.cli.ui.Application")
def test_wait_for_navigation_key_handles_eof_error(mock_app_cls: MagicMock) -> None:
    """Test wait_for_navigation_key returns 'q' on EOFError."""
    mock_app_instance = MagicMock()
    mock_app_cls.return_value = mock_app_instance
    mock_app_instance.run.side_effect = EOFError

    result = wait_for_navigation_key()

    assert result == "q"
