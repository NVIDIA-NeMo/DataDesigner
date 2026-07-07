# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.create import create_command
from data_designer.engine.storage.artifact_storage import ResumeMode

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
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
        resume=ResumeMode.NEVER,
        output_format=None,
        tui=False,
        script_args=None,
    )
