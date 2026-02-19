# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call, patch

import pytest
import typer

from data_designer.cli.controllers.generation_controller import GenerationController
from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.utils.constants import DEFAULT_DISPLAY_WIDTH

_CTRL = "data_designer.cli.controllers.generation_controller"
_DW = DEFAULT_DISPLAY_WIDTH


def _make_mock_preview_results(num_records: int) -> MagicMock:
    """Create a mock PreviewResults with the given number of records."""
    mock_results = MagicMock()
    mock_results.dataset = MagicMock()
    mock_results.dataset.__len__ = MagicMock(return_value=num_records)
    return mock_results


def _make_mock_create_results(num_records: int, base_path: str = "/output/artifacts/dataset") -> MagicMock:
    """Create a mock CreateResults with the given number of records."""
    mock_results = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=num_records)
    mock_results.load_dataset.return_value = mock_dataset
    mock_results.artifact_storage.base_dataset_path = base_path
    return mock_results


# ---------------------------------------------------------------------------
# run_preview tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_success(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test successful preview execution in non-interactive mode."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.preview.return_value = _make_mock_preview_results(5)

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=5, non_interactive=True)

    mock_load_config.assert_called_once_with("config.yaml")
    mock_dd_cls.assert_called_once()
    mock_dd.preview.assert_called_once_with(mock_builder, num_records=5)


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_custom_num_records(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test preview with a custom number of records."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.preview.return_value = _make_mock_preview_results(20)

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=20, non_interactive=True)

    mock_dd.preview.assert_called_once_with(mock_builder, num_records=20)


@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_config_load_error(mock_load_config: MagicMock) -> None:
    """Test preview exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_preview(config_source="missing.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_generation_fails(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test preview exits with code 1 when generation fails."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.preview.side_effect = RuntimeError("LLM connection failed")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_preview(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_no_records_generated(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test preview exits with code 1 when dataset is None."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = MagicMock()
    mock_results.dataset = None
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_preview(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_empty_dataset(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test preview exits with code 1 when dataset is empty."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = MagicMock()
    mock_results.dataset = MagicMock()
    mock_results.dataset.__len__ = MagicMock(return_value=0)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_preview(config_source="config.yaml", num_records=10, non_interactive=True)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_non_interactive_displays_all(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test --non-interactive displays all records without interactive browsing."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=True)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=1, display_width=_DW),
            call(index=2, display_width=_DW),
        ]
    )


@patch(f"{_CTRL}.sys")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_non_tty_stdin_falls_back_to_non_interactive(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    mock_sys: MagicMock,
) -> None:
    """Test non-TTY stdin auto-detects and falls back to non-interactive mode."""
    mock_sys.stdin.isatty.return_value = False
    mock_sys.stdout.isatty.return_value = True
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=False)

    assert mock_results.display_sample_record.call_count == 3


@patch(f"{_CTRL}.sys")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_piped_stdout_falls_back_to_non_interactive(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    mock_sys: MagicMock,
) -> None:
    """Test piped stdout (e.g. `preview cfg.yaml | head`) falls back to non-interactive."""
    mock_sys.stdin.isatty.return_value = True
    mock_sys.stdout.isatty.return_value = False
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=False)

    assert mock_results.display_sample_record.call_count == 3


@patch(f"{_CTRL}.sys")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_single_record_no_interactive(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    mock_sys: MagicMock,
) -> None:
    """Test single record is displayed directly without interactive prompt."""
    mock_sys.stdin.isatty.return_value = True
    mock_sys.stdout.isatty.return_value = True
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(1)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=1, non_interactive=False)

    mock_results.display_sample_record.assert_called_once_with(index=0, display_width=_DW)


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["n", "q"])
@patch(f"{_CTRL}.sys")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_tty_multiple_records_uses_interactive(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    mock_sys: MagicMock,
    mock_wait: MagicMock,
) -> None:
    """Test TTY with multiple records triggers interactive mode."""
    mock_sys.stdin.isatty.return_value = True
    mock_sys.stdout.isatty.return_value = True
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=False)

    assert mock_results.display_sample_record.call_count == 2
    assert mock_wait.call_count == 2


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_calls_to_report_when_analysis_present(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, mock_create_pager: MagicMock, tmp_path: Path
) -> None:
    """Test that to_report() is called only for file save (not console) when save_results=True."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_analysis = MagicMock()
    mock_results.analysis = mock_analysis
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(
        config_source="config.yaml", num_records=3, non_interactive=True, save_results=True, artifact_path=str(tmp_path)
    )

    mock_analysis.to_report.assert_called_once()
    assert mock_analysis.to_report.call_args.kwargs["save_path"].name == "report.html"


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_save_results_creates_directory_structure(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    mock_create_pager: MagicMock,
    tmp_path: Path,
) -> None:
    """Test --save-results saves dataset, report, sample records, and sample_records_browser.html."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(2)
    mock_analysis = MagicMock()
    mock_results.analysis = mock_analysis
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(
        config_source="config.yaml",
        num_records=2,
        non_interactive=True,
        save_results=True,
        artifact_path=str(tmp_path),
    )

    # Report saved to file only (no console display when save_results=True)
    mock_analysis.to_report.assert_called_once()
    report_save_path = mock_analysis.to_report.call_args.kwargs["save_path"]
    assert report_save_path.parent.parent == tmp_path
    assert report_save_path.name == "report.html"

    # Dataset saved as parquet
    mock_results.dataset.to_parquet.assert_called_once()
    parquet_path = mock_results.dataset.to_parquet.call_args[0][0]
    assert parquet_path.name == "dataset.parquet"
    assert parquet_path.parent == report_save_path.parent

    assert mock_results.display_sample_record.call_count == 2
    sample_records_dir = report_save_path.parent / "sample_records"
    for i in range(2):
        mock_results.display_sample_record.assert_any_call(
            index=i, save_path=sample_records_dir / f"record_{i}.html", theme="dark", display_width=110
        )

    # Sample records browser (pager) generated
    pager_kwargs = mock_create_pager.call_args.kwargs
    assert pager_kwargs["sample_records_dir"] == sample_records_dir
    assert pager_kwargs["num_records"] == 2
    assert "num_columns" in pager_kwargs


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_save_results_default_artifact_path(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, mock_create_pager: MagicMock
) -> None:
    """Test --save-results with no artifact_path defaults to ./artifacts."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(1)
    mock_analysis = MagicMock()
    mock_results.analysis = mock_analysis
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    with patch.object(Path, "mkdir"):
        controller.run_preview(
            config_source="config.yaml",
            num_records=1,
            non_interactive=True,
            save_results=True,
        )

    mock_analysis.to_report.assert_called_once()
    report_save_path = mock_analysis.to_report.call_args.kwargs["save_path"]
    assert report_save_path.parent.parent == Path.cwd() / "artifacts"
    mock_create_pager.assert_called_once()


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_skips_report_when_analysis_is_none(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test that to_report() is not called when analysis is None."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_results.analysis = None
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    # Implicit assertion: analysis is None (not a mock), so the code must not call
    # None.to_report(). If it does, an AttributeError propagates and the test fails.
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=True)


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_save_results_without_analysis(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, mock_create_pager: MagicMock, tmp_path: Path
) -> None:
    """Test --save-results saves dataset and sample records even when analysis is None."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(2)
    mock_results.analysis = None
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(
        config_source="config.yaml",
        num_records=2,
        non_interactive=True,
        save_results=True,
        artifact_path=str(tmp_path),
    )

    mock_results.dataset.to_parquet.assert_called_once()
    save_path_calls = [c for c in mock_results.display_sample_record.call_args_list if "save_path" in c.kwargs]
    assert len(save_path_calls) == 2


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_no_save_when_save_results_false(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test that dataset and sample records are not saved when save_results=False."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(3)
    mock_dd.preview.return_value = mock_results

    controller = GenerationController()
    controller.run_preview(config_source="config.yaml", num_records=3, non_interactive=True)

    mock_results.dataset.to_parquet.assert_not_called()
    for c in mock_results.display_sample_record.call_args_list:
        assert "save_path" not in c.kwargs


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_save_results_oserror_exits(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, mock_create_pager: MagicMock, tmp_path: Path
) -> None:
    """Test --save-results exits with code 1 when an OSError occurs."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(2)
    mock_results.analysis = None
    mock_dd.preview.return_value = mock_results
    mock_results.dataset.to_parquet.side_effect = OSError("Disk full")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_preview(
            config_source="config.yaml",
            num_records=2,
            non_interactive=True,
            save_results=True,
            artifact_path=str(tmp_path),
        )

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.create_sample_records_pager")
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_preview_save_results_non_oserror_propagates(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, mock_create_pager: MagicMock, tmp_path: Path
) -> None:
    """Test --save-results lets non-OSError exceptions propagate."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_preview_results(2)
    mock_results.analysis = None
    mock_dd.preview.return_value = mock_results
    mock_results.dataset.to_parquet.side_effect = ValueError("Unexpected error")

    controller = GenerationController()
    with pytest.raises(ValueError, match="Unexpected error"):
        controller.run_preview(
            config_source="config.yaml",
            num_records=2,
            non_interactive=True,
            save_results=True,
            artifact_path=str(tmp_path),
        )


# ---------------------------------------------------------------------------
# _browse_records_interactively unit tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["n", "n", "q"])
def test_browse_interactively_next_advances(mock_wait: MagicMock) -> None:
    """Test pressing n/enter advances to the next record."""
    mock_results = _make_mock_preview_results(5)
    controller = GenerationController()

    controller._browse_records_interactively(mock_results, 5)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=1, display_width=_DW),
            call(index=2, display_width=_DW),
        ]
    )


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["q"])
def test_browse_interactively_quit_immediately(mock_wait: MagicMock) -> None:
    """Test pressing 'q' quits after showing only the first record."""
    mock_results = _make_mock_preview_results(5)
    controller = GenerationController()

    controller._browse_records_interactively(mock_results, 5)

    mock_results.display_sample_record.assert_called_once_with(index=0, display_width=_DW)


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["n", "p", "q"])
def test_browse_interactively_previous(mock_wait: MagicMock) -> None:
    """Test 'p' navigates to the previous record."""
    mock_results = _make_mock_preview_results(5)
    controller = GenerationController()

    controller._browse_records_interactively(mock_results, 5)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=1, display_width=_DW),
            call(index=0, display_width=_DW),
        ]
    )


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["p", "q"])
def test_browse_interactively_previous_wraps_to_last(mock_wait: MagicMock) -> None:
    """Test 'p' on the first record wraps to the last record."""
    mock_results = _make_mock_preview_results(3)
    controller = GenerationController()

    controller._browse_records_interactively(mock_results, 3)

    assert mock_results.display_sample_record.call_count == 2
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=2, display_width=_DW),
        ]
    )


@patch(f"{_CTRL}.wait_for_navigation_key", side_effect=["n", "n", "n", "q"])
def test_browse_interactively_next_wraps_past_last(mock_wait: MagicMock) -> None:
    """Test n past the last record wraps back to the first."""
    mock_results = _make_mock_preview_results(3)
    controller = GenerationController()

    controller._browse_records_interactively(mock_results, 3)

    assert mock_results.display_sample_record.call_count == 4
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=1, display_width=_DW),
            call(index=2, display_width=_DW),
            call(index=0, display_width=_DW),
        ]
    )


# ---------------------------------------------------------------------------
# _display_all_records unit test
# ---------------------------------------------------------------------------


def test_display_all_records() -> None:
    """Test _display_all_records displays every record."""
    mock_results = _make_mock_preview_results(3)
    controller = GenerationController()

    controller._display_all_records(mock_results, 3)

    assert mock_results.display_sample_record.call_count == 3
    mock_results.display_sample_record.assert_has_calls(
        [
            call(index=0, display_width=_DW),
            call(index=1, display_width=_DW),
            call(index=2, display_width=_DW),
        ]
    )


# ---------------------------------------------------------------------------
# run_validate tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_validate_success(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test successful validate execution."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.validate.return_value = None

    controller = GenerationController()
    controller.run_validate(config_source="config.yaml")

    mock_load_config.assert_called_once_with("config.yaml")
    mock_dd_cls.assert_called_once()
    mock_dd.validate.assert_called_once_with(mock_builder)


@patch(f"{_CTRL}.load_config_builder")
def test_run_validate_config_load_error(mock_load_config: MagicMock) -> None:
    """Test validate exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_validate(config_source="missing.yaml")

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_validate_invalid_config(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test validate exits with code 1 when config is invalid."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.validate.side_effect = InvalidConfigError("Missing required column")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_validate(config_source="config.yaml")

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_validate_generic_exception(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test validate exits with code 1 on unexpected errors."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.validate.side_effect = RuntimeError("Unexpected error")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_validate(config_source="config.yaml")

    assert exc_info.value.exit_code == 1


# ---------------------------------------------------------------------------
# run_create tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_success(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test successful create execution with default artifact path."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results(10)

    controller = GenerationController()
    controller.run_create(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    mock_load_config.assert_called_once_with("config.yaml")
    mock_dd_cls.assert_called_once_with(artifact_path=Path.cwd() / "artifacts")
    mock_dd.create.assert_called_once_with(mock_builder, num_records=10, dataset_name="dataset")


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_custom_options(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test create with custom --num-records, --dataset-name, and --artifact-path."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results(100, "/custom/output/my_data")

    controller = GenerationController()
    controller.run_create(
        config_source="config.py",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
    )

    mock_dd_cls.assert_called_once_with(artifact_path=Path("/custom/output"))
    mock_dd.create.assert_called_once_with(mock_load_config.return_value, num_records=100, dataset_name="my_data")


@patch(f"{_CTRL}.load_config_builder")
def test_run_create_config_load_error(mock_load_config: MagicMock) -> None:
    """Test create exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_create(config_source="missing.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_creation_fails(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test create exits with code 1 when dataset creation fails."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.side_effect = RuntimeError("LLM connection failed")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_create(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_calls_to_report_when_analysis_present(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test that analysis.to_report() is called when load_analysis() returns a value."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_create_results(10)
    mock_analysis = MagicMock()
    mock_results.load_analysis.return_value = mock_analysis
    mock_dd.create.return_value = mock_results

    controller = GenerationController()
    controller.run_create(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    mock_results.load_analysis.assert_called_once()
    mock_analysis.to_report.assert_called_once()


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_skips_report_when_analysis_is_none(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test that to_report() is not called when load_analysis() returns None."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_create_results(10)
    mock_results.load_analysis.return_value = None
    mock_dd.create.return_value = mock_results

    controller = GenerationController()
    controller.run_create(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    # load_analysis() returns None, so to_report() must not be called.
    # If the code ignores the None check, an AttributeError propagates and the test fails.
    mock_results.load_analysis.assert_called_once()
