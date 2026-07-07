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
from data_designer.config.run_config import RequestAdmissionTuningConfig, RunConfig
from data_designer.config.script_params import DataDesignerScriptParams
from data_designer.config.utils.constants import DEFAULT_DISPLAY_WIDTH
from data_designer.engine.storage.artifact_storage import ResumeMode

_CTRL = "data_designer.cli.controllers.generation_controller"
_DW = DEFAULT_DISPLAY_WIDTH


def _make_mock_preview_results(num_records: int) -> MagicMock:
    """Create a mock PreviewResults with the given number of records."""
    mock_results = MagicMock()
    mock_results.dataset = MagicMock()
    mock_results.dataset.__len__ = MagicMock(return_value=num_records)
    return mock_results


def _make_mock_create_results(num_records: int = 0, base_path: str = "/output/artifacts/dataset") -> MagicMock:
    """Create a mock DatasetCreationResults."""
    mock_results = MagicMock()
    mock_results.count_records.return_value = num_records
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
    controller.run_preview(
        config_source="config.py",
        num_records=5,
        non_interactive=True,
        script_args=["--seed-path", "seed.parquet"],
    )

    mock_load_config.assert_called_once_with(
        "config.py",
        DataDesignerScriptParams(argv=("--seed-path", "seed.parquet")),
    )
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
    controller.run_validate(config_source="config.py", script_args=["--seed-path", "seed.parquet"])

    mock_load_config.assert_called_once_with(
        "config.py",
        DataDesignerScriptParams(argv=("--seed-path", "seed.parquet")),
    )
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
# run_check_models tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_check_models_success(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test successful check_models execution delegates to DataDesigner.check_models."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.check_models.return_value = None

    controller = GenerationController()
    controller.run_check_models(config_source="config.py", script_args=["--seed-path", "seed.parquet"])

    mock_load_config.assert_called_once_with(
        "config.py",
        DataDesignerScriptParams(argv=("--seed-path", "seed.parquet")),
    )
    mock_dd_cls.assert_called_once()
    mock_dd.check_models.assert_called_once_with(mock_builder)


@patch(f"{_CTRL}.load_config_builder")
def test_run_check_models_config_load_error(mock_load_config: MagicMock) -> None:
    """check_models exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_check_models(config_source="missing.yaml")

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_check_models_health_check_failure(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """check_models exits with code 1 when a probe fails with a generic exception."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.check_models.side_effect = RuntimeError("auth failed")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_check_models(config_source="config.yaml")

    assert exc_info.value.exit_code == 1


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_check_models_typed_error_includes_class_name(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Typed engine errors exit 1 and surface the error class name to the user.

    Without this, an authentication failure and a connection failure look identical
    on the terminal, defeating the purpose of typed engine errors.
    """
    from data_designer.engine.models.errors import ModelAuthenticationError

    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.check_models.side_effect = ModelAuthenticationError("bad creds")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_check_models(config_source="config.yaml")

    assert exc_info.value.exit_code == 1
    captured = capsys.readouterr()
    assert "ModelAuthenticationError" in captured.out
    assert "bad creds" in captured.out


# ---------------------------------------------------------------------------
# run_create tests
# ---------------------------------------------------------------------------


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_success(
    mock_load_config: MagicMock, mock_dd_cls: MagicMock, capsys: pytest.CaptureFixture[str]
) -> None:
    """Test successful create execution with default artifact path."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results()

    controller = GenerationController()
    controller.run_create(
        config_source="config.py",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        script_args=["--seed-path", "seed.parquet"],
    )

    mock_load_config.assert_called_once_with(
        "config.py",
        DataDesignerScriptParams(argv=("--seed-path", "seed.parquet")),
    )
    mock_dd_cls.assert_called_once_with(artifact_path=Path.cwd() / "artifacts")
    mock_dd.create.assert_called_once_with(
        mock_builder, num_records=10, dataset_name="dataset", resume=ResumeMode.NEVER
    )
    mock_dd.set_run_config.assert_not_called()
    assert "Run config:" not in capsys.readouterr().out


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_custom_options(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """Test create with custom --num-records, --dataset-name, and --artifact-path."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results(base_path="/custom/output/my_data")

    controller = GenerationController()
    controller.run_create(
        config_source="config.py",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
    )

    mock_dd_cls.assert_called_once_with(artifact_path=Path("/custom/output"))
    mock_dd.create.assert_called_once_with(
        mock_load_config.return_value, num_records=100, dataset_name="my_data", resume=ResumeMode.NEVER
    )


@pytest.mark.parametrize("tui", [True, False])
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_applies_tui_override(mock_load_config: MagicMock, mock_dd_cls: MagicMock, tui: bool) -> None:
    """run_create applies explicit --tui/--no-tui override to RunConfig."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd.run_config = RunConfig(display_tui=not tui)
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results()

    controller = GenerationController()
    controller.run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        tui=tui,
    )

    mock_dd.set_run_config.assert_called_once()
    assert mock_dd.set_run_config.call_args.args[0].display_tui is tui
    mock_dd.create.assert_called_once_with(
        mock_load_config.return_value, num_records=10, dataset_name="dataset", resume=ResumeMode.NEVER
    )


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_run_config")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_applies_run_config_precedence(
    mock_load_config: MagicMock,
    mock_load_run_config: MagicMock,
    mock_dd_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Runtime YAML overlays the baseline, then explicit TUI wins."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_run_config.return_value = RunConfig.model_validate(
        {
            "buffer_size": 1000,
            "disable_early_shutdown": True,
            "display_tui": True,
            "preserve_dropped_columns": False,
        }
    )
    mock_dd = MagicMock()
    mock_dd.run_config = RunConfig(buffer_size=64, progress_interval=11.0, display_tui=True)
    mock_dd.create.return_value = _make_mock_create_results()
    mock_dd_cls.return_value = mock_dd

    controller = GenerationController()
    controller.run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        run_config_source="run.yaml",
        tui=False,
    )

    mock_load_run_config.assert_called_once_with("run.yaml")
    mock_dd.set_run_config.assert_called_once()
    effective = mock_dd.set_run_config.call_args.args[0]
    assert effective.buffer_size == 1000
    assert effective.progress_interval == 11.0
    assert effective.preserve_dropped_columns is False
    assert effective.shutdown_error_rate == 1.0
    assert effective.display_tui is False
    assert [item[0] for item in mock_dd.method_calls] == ["set_run_config", "create"]
    assert "Run config: run.yaml" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("request_admission", "expected"),
    [
        (
            {"additive_increase_step": 3},
            RequestAdmissionTuningConfig(additive_increase_step=3),
        ),
        (None, None),
    ],
)
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_run_config")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_replaces_nested_run_config(
    mock_load_config: MagicMock,
    mock_load_run_config: MagicMock,
    mock_dd_cls: MagicMock,
    request_admission: dict[str, int] | None,
    expected: RequestAdmissionTuningConfig | None,
) -> None:
    """Supplying request_admission replaces, rather than deep-merges, the baseline object."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_run_config.return_value = RunConfig.model_validate({"request_admission": request_admission})
    mock_dd = MagicMock()
    mock_dd.run_config = RunConfig(
        request_admission=RequestAdmissionTuningConfig(
            multiplicative_decrease_factor=0.5,
            additive_increase_step=9,
            successes_until_increase=99,
            cooldown_seconds=9.0,
            startup_ramp_seconds=9.0,
        )
    )
    mock_dd.create.return_value = _make_mock_create_results()
    mock_dd_cls.return_value = mock_dd

    GenerationController().run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        run_config_source="run.yml",
    )

    effective = mock_dd.set_run_config.call_args.args[0]
    assert effective.request_admission == expected


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_run_config")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_applies_empty_run_config(
    mock_load_config: MagicMock, mock_load_run_config: MagicMock, mock_dd_cls: MagicMock
) -> None:
    """An explicitly supplied empty mapping still applies the active baseline once."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_run_config.return_value = RunConfig.model_validate({})
    mock_dd = MagicMock()
    mock_dd.run_config = RunConfig(buffer_size=42)
    mock_dd.create.return_value = _make_mock_create_results()
    mock_dd_cls.return_value = mock_dd

    GenerationController().run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        run_config_source="empty.yaml",
    )

    mock_dd.set_run_config.assert_called_once_with(RunConfig(buffer_size=42))


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_run_config")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_run_config_load_error(
    mock_load_config: MagicMock,
    mock_load_run_config: MagicMock,
    mock_dd_cls: MagicMock,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Invalid runtime YAML exits before DataDesigner construction."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_run_config.side_effect = ConfigLoadError(
        "Failed to load run config from 'bad.yaml': buffer_size: Input should be greater than 0"
    )

    with pytest.raises(typer.Exit) as exc_info:
        GenerationController().run_create(
            config_source="config.yaml",
            num_records=10,
            dataset_name="dataset",
            artifact_path=None,
            run_config_source="bad.yaml",
        )

    assert exc_info.value.exit_code == 1
    mock_dd_cls.assert_not_called()
    output = capsys.readouterr().out
    assert "bad.yaml" in output
    assert "buffer_size" in output
    assert "Dataset creation failed" not in output


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
    mock_results = _make_mock_create_results()
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
    mock_results = _make_mock_create_results()
    mock_results.load_analysis.return_value = None
    mock_dd.create.return_value = mock_results

    controller = GenerationController()
    controller.run_create(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    # load_analysis() returns None, so to_report() must not be called.
    # If the code ignores the None check, an AttributeError propagates and the test fails.
    mock_results.load_analysis.assert_called_once()


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_passes_resume_always(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """run_create forwards resume=ALWAYS to DataDesigner.create()."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results()

    controller = GenerationController()
    controller.run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.ALWAYS,
    )

    mock_dd.create.assert_called_once_with(
        mock_load_config.return_value, num_records=10, dataset_name="dataset", resume=ResumeMode.ALWAYS
    )


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_with_output_format_happy_path(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """export() is called with the dataset-name-derived path when --output-format is given."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_create_results(5, "/output/artifacts/my_data")
    mock_dd.create.return_value = mock_results

    controller = GenerationController()
    controller.run_create(
        config_source="config.yaml",
        num_records=5,
        dataset_name="my_data",
        artifact_path=None,
        output_format="jsonl",
    )

    mock_results.export.assert_called_once_with(
        Path("/output/artifacts/my_data") / "my_data.jsonl",
    )


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_passes_resume_if_possible(mock_load_config: MagicMock, mock_dd_cls: MagicMock) -> None:
    """run_create forwards resume=IF_POSSIBLE to DataDesigner.create()."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_dd.create.return_value = _make_mock_create_results()

    controller = GenerationController()
    controller.run_create(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.IF_POSSIBLE,
    )

    mock_dd.create.assert_called_once_with(
        mock_load_config.return_value, num_records=10, dataset_name="dataset", resume=ResumeMode.IF_POSSIBLE
    )


@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_run_create_export_failure_exits(mock_load_config: MagicMock, mock_dd_cls: MagicMock, tmp_path: Path) -> None:
    """If export() raises, run_create cleans up the partial file and exits with code 1."""
    mock_load_config.return_value = MagicMock(spec=DataDesignerConfigBuilder)
    mock_dd = MagicMock()
    mock_dd_cls.return_value = mock_dd
    mock_results = _make_mock_create_results(5, str(tmp_path))
    mock_results.export.side_effect = RuntimeError("disk full")
    mock_dd.create.return_value = mock_results

    # Create a partial file to verify it gets cleaned up.
    partial_file = tmp_path / "dataset.csv"
    partial_file.write_text("partial")

    controller = GenerationController()
    with pytest.raises(typer.Exit) as exc_info:
        controller.run_create(
            config_source="config.yaml",
            num_records=5,
            dataset_name="dataset",
            artifact_path=None,
            output_format="csv",
        )
    assert exc_info.value.exit_code == 1
    assert not partial_file.exists()
