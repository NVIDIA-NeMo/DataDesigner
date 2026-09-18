# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from data_designer.cli.commands.create import create_command
from data_designer.cli.main import app
from data_designer.config.run_config import RunConfig
from data_designer.engine.storage.artifact_storage import ResumeMode

runner = CliRunner()
_CTRL = "data_designer.cli.controllers.generation_controller"

# ---------------------------------------------------------------------------
# create_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command delegates to GenerationController.run_create."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=None,
    )

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_passes_custom_options(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command passes custom options to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.py",
        run_config_source="run-config.yaml",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=["--seed-path", "seed.parquet"],
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.py",
        run_config_source="run-config.yaml",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=["--seed-path", "seed.parquet"],
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_default_artifact_path_is_none(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command passes artifact_path=None when not specified."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=5,
        dataset_name="ds",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=None,
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=5,
        dataset_name="ds",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_passes_resume_always(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command forwards --resume always to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.ALWAYS,
        output_format=None,
        tui=None,
        script_args=None,
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.ALWAYS,
        output_format=None,
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_passes_resume_if_possible(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command forwards --resume if_possible to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.IF_POSSIBLE,
        output_format=None,
        tui=None,
        script_args=None,
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.IF_POSSIBLE,
        output_format=None,
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_passes_output_format(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command forwards --output-format to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format="jsonl",
        tui=None,
        script_args=None,
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format="jsonl",
        tui=None,
        script_args=None,
    )


@patch("data_designer.cli.commands.create.GenerationController")
def test_create_command_passes_tui_override(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command forwards explicit TUI override."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=False,
        script_args=None,
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        run_config_source=None,
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=False,
        script_args=None,
    )


# ---------------------------------------------------------------------------
# create --run-config overlay tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("yaml_text", "expected_rate", "expected_effective_rate"),
    [
        ("disable_early_shutdown: true\n", 0.3, 1.0),
        ("disable_early_shutdown: true\nshutdown_error_rate: 0.45\n", 0.45, 1.0),
        ("shutdown_error_rate: 0.45\n", 0.45, 0.45),
    ],
    ids=["flag-only-keeps-baseline-rate", "flag-with-explicit-rate", "rate-only"],
)
@patch(f"{_CTRL}.DataDesigner")
@patch(f"{_CTRL}.load_config_builder")
def test_create_run_config_overlay_preserves_shutdown_error_rate(
    mock_load_config: MagicMock,
    mock_dd_cls: MagicMock,
    tmp_path: Path,
    yaml_text: str,
    expected_rate: float,
    expected_effective_rate: float,
) -> None:
    """`create --run-config` overlays the YAML file without discarding the baseline error rate."""
    run_config_file = tmp_path / "run-config.yaml"
    run_config_file.write_text(yaml_text)

    mock_load_config.return_value = MagicMock()
    mock_results = MagicMock()
    mock_results.count_records.return_value = 10
    mock_results.load_analysis.return_value = None
    mock_dd = MagicMock()
    mock_dd.run_config = RunConfig(shutdown_error_rate=0.3)
    mock_dd.create.return_value = mock_results
    mock_dd_cls.return_value = mock_dd

    result = runner.invoke(app, ["create", "dataset.yaml", "--run-config", str(run_config_file)])

    assert result.exit_code == 0, result.output
    effective = mock_dd.set_run_config.call_args.args[0]
    assert effective.shutdown_error_rate == expected_rate
    assert effective.effective_shutdown_error_rate == expected_effective_rate
