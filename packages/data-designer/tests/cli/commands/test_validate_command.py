# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock, patch

import pytest
import typer

from data_designer.cli.commands.validate import validate_command
from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.validate.load_config_builder")
def test_validate_command_success(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test successful validate command execution."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.validate.return_value = None

    validate_command(config_source="config.yaml")

    mock_load_config.assert_called_once_with("config.yaml")
    mock_data_designer_cls.assert_called_once()
    mock_dd_instance.validate.assert_called_once_with(mock_builder)


@patch("data_designer.cli.commands.validate.load_config_builder")
def test_validate_command_config_load_error(mock_load_config: MagicMock) -> None:
    """Test validate command exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    with pytest.raises(typer.Exit) as exc_info:
        validate_command(config_source="missing.yaml")

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.validate.load_config_builder")
def test_validate_command_invalid_config(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test validate command exits with code 1 when config is invalid."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.validate.side_effect = InvalidConfigError("Missing required column")

    with pytest.raises(typer.Exit) as exc_info:
        validate_command(config_source="config.yaml")

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.validate.load_config_builder")
def test_validate_command_generic_exception(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test validate command exits with code 1 on unexpected errors."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.validate.side_effect = RuntimeError("Unexpected error")

    with pytest.raises(typer.Exit) as exc_info:
        validate_command(config_source="config.yaml")

    assert exc_info.value.exit_code == 1
