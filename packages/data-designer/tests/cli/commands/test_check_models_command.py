# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

from data_designer.cli.commands.check_models import check_models_command

# ---------------------------------------------------------------------------
# check_models_command delegation tests
# ---------------------------------------------------------------------------


@patch("data_designer.cli.commands.check_models.GenerationController")
def test_check_models_command_delegates_to_controller(mock_ctrl_cls: MagicMock) -> None:
    """check_models_command delegates to GenerationController.run_check_models."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    check_models_command(config_source="config.yaml", script_args=None)

    mock_ctrl_cls.assert_called_once()
    mock_ctrl.run_check_models.assert_called_once_with(config_source="config.yaml", script_args=None)


@patch("data_designer.cli.commands.check_models.GenerationController")
def test_check_models_command_passes_python_module_source(mock_ctrl_cls: MagicMock) -> None:
    """check_models_command passes a .py source to the controller."""
    mock_ctrl = MagicMock()
    mock_ctrl_cls.return_value = mock_ctrl

    check_models_command(config_source="my_config.py", script_args=["--seed-path", "seed.parquet"])

    mock_ctrl.run_check_models.assert_called_once_with(
        config_source="my_config.py",
        script_args=["--seed-path", "seed.parquet"],
    )
