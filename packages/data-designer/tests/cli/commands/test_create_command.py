# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import typer

from data_designer.cli.commands.create import create_command
from data_designer.cli.utils.config_loader import ConfigLoadError
from data_designer.config.config_builder import DataDesignerConfigBuilder


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.create.load_config_builder")
def test_create_command_success(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test successful create command execution."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance

    mock_results = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=10)
    mock_results.load_dataset.return_value = mock_dataset
    mock_results.artifact_storage.base_dataset_path = "/output/artifacts/dataset"
    mock_dd_instance.create.return_value = mock_results

    create_command(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    mock_load_config.assert_called_once_with("config.yaml")
    mock_data_designer_cls.assert_called_once_with(artifact_path=Path.cwd() / "artifacts")
    mock_dd_instance.create.assert_called_once_with(mock_builder, num_records=10, dataset_name="dataset")


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.create.load_config_builder")
def test_create_command_custom_options(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test create command with custom --num-records, --dataset-name, and --artifact-path."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance

    mock_results = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=100)
    mock_results.load_dataset.return_value = mock_dataset
    mock_results.artifact_storage.base_dataset_path = "/custom/output/my_data"
    mock_dd_instance.create.return_value = mock_results

    create_command(
        config_source="config.py",
        num_records=100,
        dataset_name="my_data",
        artifact_path="/custom/output",
    )

    mock_data_designer_cls.assert_called_once_with(artifact_path=Path("/custom/output"))
    mock_dd_instance.create.assert_called_once_with(mock_builder, num_records=100, dataset_name="my_data")


@patch("data_designer.cli.commands.create.load_config_builder")
def test_create_command_config_load_error(mock_load_config: MagicMock) -> None:
    """Test create command exits with code 1 when config fails to load."""
    mock_load_config.side_effect = ConfigLoadError("File not found")

    with pytest.raises(typer.Exit) as exc_info:
        create_command(config_source="missing.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.create.load_config_builder")
def test_create_command_creation_fails(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test create command exits with code 1 when dataset creation fails."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance
    mock_dd_instance.create.side_effect = RuntimeError("LLM connection failed")

    with pytest.raises(typer.Exit) as exc_info:
        create_command(config_source="config.yaml", num_records=10, dataset_name="dataset", artifact_path=None)

    assert exc_info.value.exit_code == 1


@patch("data_designer.interface.DataDesigner")
@patch("data_designer.cli.commands.create.load_config_builder")
def test_create_command_default_artifact_path(
    mock_load_config: MagicMock,
    mock_data_designer_cls: MagicMock,
) -> None:
    """Test that the default artifact path is ./artifacts when not specified."""
    mock_builder = MagicMock(spec=DataDesignerConfigBuilder)
    mock_load_config.return_value = mock_builder

    mock_dd_instance = MagicMock()
    mock_data_designer_cls.return_value = mock_dd_instance

    mock_results = MagicMock()
    mock_dataset = MagicMock()
    mock_dataset.__len__ = MagicMock(return_value=5)
    mock_results.load_dataset.return_value = mock_dataset
    mock_results.artifact_storage.base_dataset_path = str(Path.cwd() / "artifacts" / "dataset")
    mock_dd_instance.create.return_value = mock_results

    create_command(config_source="config.yaml", num_records=5, dataset_name="dataset", artifact_path=None)

    expected_path = Path.cwd() / "artifacts"
    mock_data_designer_cls.assert_called_once_with(artifact_path=expected_path)
