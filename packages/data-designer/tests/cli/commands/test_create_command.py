# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.create import create_command

# ---------------------------------------------------------------------------
# create_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.controllers.generation_controller.GenerationController")
def test_create_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command delegates to GenerationController.run_create."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=10,
        dataset_name="dataset",
        artifact_path=None,
    )


@patch("data_designer.cli.controllers.generation_controller.GenerationController")
def test_create_command_passes_custom_options(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command passes custom options to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(
        config_source="config.py",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
    )

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.py",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
    )


@patch("data_designer.cli.controllers.generation_controller.GenerationController")
def test_create_command_default_artifact_path_is_none(mock_ctrl_cls: MagicMock) -> None:
    """Test create_command passes artifact_path=None when not specified."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    create_command(config_source="config.yaml", num_records=5, dataset_name="ds", artifact_path=None)

    mock_ctrl.run_create.assert_called_once_with(
        config_source="config.yaml",
        num_records=5,
        dataset_name="ds",
        artifact_path=None,
    )
