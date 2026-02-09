# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, call, patch

import pytest
import typer

from data_designer.cli.commands.preview import (
    _browse_records_interactively,
    _display_all_records,
    preview_command,
)
from data_designer.cli.ui import wait_for_navigation_key
from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.config.config_builder import DataDesignerConfigBuilder


def _make_mock_results(num_records: int) -> MagicMock:
    """Create a mock results object with the given number of records."""
    mock_results = MagicMock()
    mock_results.dataset = MagicMock()
    mock_results.dataset.__len__ = MagicMock(return_value=num_records)
    mock_results.dataset.columns = ["col_a", "col_b"]
    mock_results.dataset.iloc.__getitem__ = MagicMock(return_value=MagicMock())
    return mock_results


# ---------------------------------------------------------------------------
# preview_command integration tests
# ---------------------------------------------------------------------------


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_success(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test successful preview command execution."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.preview.return_value = _make_mock_results(5)

    preview_command(config_source="config.yaml", num_records=5, non_interactive=True)

    mock_load_config.assert_called_once_with("config.yaml")
    mock_data_designer_cls.assert_called_once()
    mock_dd_instance.preview.assert_called_once_with(mock_builder, num_records=5)


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_custom_num_records(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test preview command with custom --num-records."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.preview.return_value = _make_mock_results(20)

    preview_command(config_source="config.yaml", num_records=20, non_interactive=True)

    mock_dd_instance.preview.assert_called_once_with(mock_builder, num_records=20)


@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_config_load_error(mock_load_config: MagicMock) -> None:
    """Test preview command exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    with pytest.raises(typer.Exit) as exc_info:
        preview_command(config_source="missing.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_generation_fails(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test preview command exits with code 1 when preview generation fails."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.preview.side_effect = RuntimeError("LLM connection failed")

    with pytest.raises(typer.Exit) as exc_info:
        preview_command(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_no_records_generated(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test preview command exits with code 1 when no records are generated."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance

    mock_results = MagicMock()
    mock_results.dataset = None
    mock_dd_instance.preview.return_value = mock_results

    with pytest.raises(typer.Exit) as exc_info:
        preview_command(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_empty_dataset(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test preview command exits with code 1 when dataset is empty."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance

    mock_results = MagicMock()
    mock_results.dataset = MagicMock()
    mock_results.dataset.__len__ = MagicMock(return_value=0)
    mock_dd_instance.preview.return_value = mock_results

    with pytest.raises(typer.Exit) as exc_info:
        preview_command(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_non_interactive_flag(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test --non-interactive flag displays all records without interactive browsing."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_results = _make_mock_results(3)
    mock_dd_instance.preview.return_value = mock_results

    preview_command(config_source="config.yaml", num_records=3, non_interactive=True)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_any_call(index=0)
    mock_results.display_sample_record.assert_any_call(index=1)
    mock_results.display_sample_record.assert_any_call(index=2)


@patch("data_designer.cli.commands.preview.sys")
@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_non_tty_falls_back_to_non_interactive(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
    mock_sys: MagicMock,
) -> None:
    """Test non-TTY stdin auto-detects and falls back to non-interactive mode."""
    mock_sys.stdin.isatty.return_value = False

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_results = _make_mock_results(3)
    mock_dd_instance.preview.return_value = mock_results

    preview_command(config_source="config.yaml", num_records=3, non_interactive=False)

    assert mock_results.display_sample_record.call_count == 3


@patch("data_designer.cli.commands.preview.sys")
@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_single_record_no_interactive(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
    mock_sys: MagicMock,
) -> None:
    """Test single record is displayed directly without interactive prompt."""
    mock_sys.stdin.isatty.return_value = True

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_results = _make_mock_results(1)
    mock_dd_instance.preview.return_value = mock_results

    preview_command(config_source="config.yaml", num_records=1, non_interactive=False)

    mock_results.display_sample_record.assert_called_once_with(index=0)


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "q"])
@patch("data_designer.cli.commands.preview.sys")
@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.preview.load_config_builder")
def test_preview_command_tty_multiple_records_uses_interactive(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
    mock_sys: MagicMock,
    mock_wait: MagicMock,
) -> None:
    """Test TTY with multiple records triggers interactive mode."""
    mock_sys.stdin.isatty.return_value = True

    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_results = _make_mock_results(3)
    mock_dd_instance.preview.return_value = mock_results

    preview_command(config_source="config.yaml", num_records=3, non_interactive=False)

    # First record shown immediately, then "n" advances to second, then "q" quits
    assert mock_results.display_sample_record.call_count == 2
    assert mock_wait.call_count == 2


# ---------------------------------------------------------------------------
# _browse_records_interactively unit tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "n", "q"])
def test_browse_interactively_enter_advances(mock_wait: MagicMock) -> None:
    """Test pressing n/enter advances to the next record."""
    mock_results = _make_mock_results(5)

    _browse_records_interactively(mock_results, 5)

    # Record 0 shown immediately, then n -> record 1, n -> record 2, then q
    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1), call(index=2)])


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["q"])
def test_browse_interactively_quit_immediately(mock_wait: MagicMock) -> None:
    """Test pressing 'q' on the first prompt quits after showing only the first record."""
    mock_results = _make_mock_results(5)

    _browse_records_interactively(mock_results, 5)

    mock_results.display_sample_record.assert_called_once_with(index=0)


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "p", "q"])
def test_browse_interactively_previous(mock_wait: MagicMock) -> None:
    """Test 'p' navigates to the previous record."""
    mock_results = _make_mock_results(5)

    _browse_records_interactively(mock_results, 5)

    # Record 0 shown immediately, n -> record 1, p -> record 0, then q
    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1), call(index=0)])


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["p", "q"])
def test_browse_interactively_previous_wraps_to_last(mock_wait: MagicMock) -> None:
    """Test 'p' on the first record wraps to the last record."""
    mock_results = _make_mock_results(3)

    _browse_records_interactively(mock_results, 3)

    # Record 0 shown immediately, p wraps -> record 2, then q
    assert mock_results.display_sample_record.call_count == 2
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=2)])


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "n", "q"])
def test_browse_interactively_enter_wraps_to_first(mock_wait: MagicMock) -> None:
    """Test n on the last record wraps to the first record."""
    mock_results = _make_mock_results(3)

    _browse_records_interactively(mock_results, 3)

    # Record 0 shown immediately, n -> record 1, n -> record 2, then q
    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1), call(index=2)])


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "n", "n", "q"])
def test_browse_interactively_enter_wraps_past_last(mock_wait: MagicMock) -> None:
    """Test n past the last record wraps back to the first."""
    mock_results = _make_mock_results(3)

    _browse_records_interactively(mock_results, 3)

    # Record 0, n -> 1, n -> 2, n -> wraps to 0, then q
    assert mock_results.display_sample_record.call_count == 4
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1), call(index=2), call(index=0)])


@patch("data_designer.cli.commands.preview.wait_for_navigation_key", side_effect=["n", "q"])
def test_browse_interactively_n_key_advances(mock_wait: MagicMock) -> None:
    """Test 'n' key advances to the next record."""
    mock_results = _make_mock_results(5)

    _browse_records_interactively(mock_results, 5)

    # Record 0 shown immediately, n -> record 1, then q
    assert mock_results.display_sample_record.call_count == 2
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1)])


# ---------------------------------------------------------------------------
# wait_for_navigation_key unit tests
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


# ---------------------------------------------------------------------------
# _display_all_records unit test
# ---------------------------------------------------------------------------


def test_display_all_records() -> None:
    """Test _display_all_records displays every record."""
    mock_results = _make_mock_results(3)

    _display_all_records(mock_results, 3)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls([call(index=0), call(index=1), call(index=2)])
