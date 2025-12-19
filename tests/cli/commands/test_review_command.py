# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from typer.testing import CliRunner

from data_designer.cli.main import app


@pytest.fixture
def sample_dataset(tmp_path: Path) -> Path:
    """Create a sample dataset for testing.

    Args:
        tmp_path: Pytest temporary directory fixture

    Returns:
        Path to the test dataset
    """
    df = pd.DataFrame({"col1": [1, 2, 3]})
    dataset_path = tmp_path / "test.parquet"
    df.to_parquet(dataset_path)
    return dataset_path


def test_review_command_missing_dataset_arg() -> None:
    """Test review command without required dataset argument."""
    runner = CliRunner()
    result = runner.invoke(app, ["review"])

    # Should fail with missing required option
    assert result.exit_code != 0


def test_review_command_nonexistent_dataset() -> None:
    """Test review command with nonexistent dataset path."""
    runner = CliRunner()
    result = runner.invoke(app, ["review", "--dataset", "nonexistent.parquet"])

    # Should fail with file not found error
    assert result.exit_code != 0


@patch("data_designer.cli.commands.review.ReviewController")
def test_review_command_success(mock_controller_class: MagicMock, sample_dataset: Path) -> None:
    """Test successful review command execution.

    Args:
        mock_controller_class: Mocked ReviewController class
        sample_dataset: Path to sample dataset
    """
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    runner = CliRunner()
    runner.invoke(app, ["review", "--dataset", str(sample_dataset)])

    # Verify controller was instantiated with correct args
    mock_controller_class.assert_called_once()
    call_kwargs = mock_controller_class.call_args[1]
    assert "dataset_path" in call_kwargs
    assert call_kwargs["port"] == 8501
    assert call_kwargs["host"] == "localhost"
    assert call_kwargs["reviewer"] == "default"

    # Verify run() was called
    mock_controller.run.assert_called_once()


@patch("data_designer.cli.commands.review.ReviewController")
def test_review_command_custom_port(mock_controller_class: MagicMock, sample_dataset: Path) -> None:
    """Test review command with custom port.

    Args:
        mock_controller_class: Mocked ReviewController class
        sample_dataset: Path to sample dataset
    """
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    runner = CliRunner()
    runner.invoke(app, ["review", "--dataset", str(sample_dataset), "--port", "8080"])

    call_kwargs = mock_controller_class.call_args[1]
    assert call_kwargs["port"] == 8080


@patch("data_designer.cli.commands.review.ReviewController")
def test_review_command_custom_reviewer(mock_controller_class: MagicMock, sample_dataset: Path) -> None:
    """Test review command with custom reviewer name.

    Args:
        mock_controller_class: Mocked ReviewController class
        sample_dataset: Path to sample dataset
    """
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    runner = CliRunner()
    runner.invoke(app, ["review", "--dataset", str(sample_dataset), "--reviewer", "john_doe"])

    call_kwargs = mock_controller_class.call_args[1]
    assert call_kwargs["reviewer"] == "john_doe"


@patch("data_designer.cli.commands.review.ReviewController")
def test_review_command_custom_host(mock_controller_class: MagicMock, sample_dataset: Path) -> None:
    """Test review command with custom host.

    Args:
        mock_controller_class: Mocked ReviewController class
        sample_dataset: Path to sample dataset
    """
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    runner = CliRunner()
    runner.invoke(app, ["review", "--dataset", str(sample_dataset), "--host", "0.0.0.0"])

    call_kwargs = mock_controller_class.call_args[1]
    assert call_kwargs["host"] == "0.0.0.0"


@patch("data_designer.cli.commands.review.ReviewController")
def test_review_command_all_options(mock_controller_class: MagicMock, sample_dataset: Path) -> None:
    """Test review command with all custom options.

    Args:
        mock_controller_class: Mocked ReviewController class
        sample_dataset: Path to sample dataset
    """
    mock_controller = MagicMock()
    mock_controller_class.return_value = mock_controller

    runner = CliRunner()
    runner.invoke(
        app,
        [
            "review",
            "--dataset",
            str(sample_dataset),
            "--port",
            "9000",
            "--host",
            "127.0.0.1",
            "--reviewer",
            "test_user",
        ],
    )

    call_kwargs = mock_controller_class.call_args[1]
    assert call_kwargs["port"] == 9000
    assert call_kwargs["host"] == "127.0.0.1"
    assert call_kwargs["reviewer"] == "test_user"
