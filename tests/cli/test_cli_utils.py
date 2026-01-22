# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.utils import (
    check_ngc_cli_available,
    get_ngc_version,
    validate_numeric_range,
    validate_url,
)
from data_designer.config.errors import InvalidConfigError, InvalidFileFormatError, InvalidFilePathError
from data_designer.config.utils.io_helpers import ensure_config_dir_exists, load_config_file, save_config_file


def test_ensure_config_dir_exists(tmp_path: Path) -> None:
    """Test creating config directory."""
    test_dir = tmp_path / "test_config"
    assert not test_dir.exists()

    ensure_config_dir_exists(test_dir)
    assert test_dir.exists()
    assert test_dir.is_dir()


def test_save_and_load_config_file(tmp_path: Path) -> None:
    """Test saving and loading config files."""
    config_file = tmp_path / "test_config.yaml"
    test_config = {
        "providers": [{"name": "test", "endpoint": "https://test.com", "provider_type": "openai"}],
        "default": "test",
    }

    # Save config
    save_config_file(config_file, test_config)
    assert config_file.exists()

    # Load config
    loaded_config = load_config_file(config_file)
    assert loaded_config == test_config


def test_load_config_file_not_found(tmp_path: Path) -> None:
    """Test loading non-existent file."""
    config_file = tmp_path / "nonexistent.yaml"

    with pytest.raises(InvalidFilePathError):
        load_config_file(config_file)


def test_load_config_file_invalid_yaml(tmp_path: Path) -> None:
    """Test loading invalid YAML."""
    config_file = tmp_path / "invalid.yaml"
    config_file.write_text("invalid: yaml: content:\n  - unmatched")

    with pytest.raises(InvalidFileFormatError):
        load_config_file(config_file)


def test_load_config_file_empty(tmp_path: Path) -> None:
    """Test loading empty file."""
    config_file = tmp_path / "empty.yaml"
    config_file.write_text("")

    with pytest.raises(InvalidConfigError):
        load_config_file(config_file)


def test_validate_url() -> None:
    """Test URL validation."""
    # Valid URLs
    assert validate_url("https://api.example.com/v1")
    assert validate_url("http://localhost:8000")
    assert validate_url("https://nvidia.com")

    # Invalid URLs
    assert not validate_url("")
    assert not validate_url("not-a-url")
    assert not validate_url("ftp://example.com")
    assert not validate_url("https://")
    assert not validate_url("http://")


def test_validate_numeric_range() -> None:
    """Test numeric range validation."""
    # Valid values
    is_valid, value = validate_numeric_range("0.5", 0.0, 1.0)
    assert is_valid
    assert value == 0.5

    is_valid, value = validate_numeric_range("0", 0.0, 1.0)
    assert is_valid
    assert value == 0.0

    is_valid, value = validate_numeric_range("1.0", 0.0, 1.0)
    assert is_valid
    assert value == 1.0

    # Invalid values - out of range
    is_valid, value = validate_numeric_range("-0.1", 0.0, 1.0)
    assert not is_valid
    assert value is None

    is_valid, value = validate_numeric_range("1.1", 0.0, 1.0)
    assert not is_valid
    assert value is None

    # Invalid values - not numeric
    is_valid, value = validate_numeric_range("abc", 0.0, 1.0)
    assert not is_valid
    assert value is None


@patch("data_designer.cli.utils.shutil.which")
@patch("data_designer.cli.utils.get_ngc_version")
def test_check_ngc_cli_available_returns_true(mock_get_version: MagicMock, mock_which: MagicMock) -> None:
    """Test NGC CLI availability check when NGC CLI is installed."""
    mock_which.return_value = "/usr/local/bin/ngc"
    mock_get_version.return_value = "NGC CLI 3.41.4"

    assert check_ngc_cli_available() is True
    mock_which.assert_called_once_with("ngc")


@patch("data_designer.cli.utils.shutil.which")
def test_check_ngc_cli_available_returns_false(mock_which: MagicMock) -> None:
    """Test NGC CLI availability check when NGC CLI is not installed."""
    mock_which.return_value = None

    assert check_ngc_cli_available() is False
    mock_which.assert_called_once_with("ngc")


@patch("data_designer.cli.utils.subprocess.run")
def test_get_ngc_version_success(mock_run: MagicMock) -> None:
    """Test getting NGC CLI version successfully."""
    mock_result = MagicMock()
    mock_result.stdout = "NGC CLI 3.41.4\n"
    mock_run.return_value = mock_result

    version = get_ngc_version()

    assert version == "NGC CLI 3.41.4"
    mock_run.assert_called_once_with(
        ["ngc", "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )


@patch("data_designer.cli.utils.subprocess.run")
def test_get_ngc_version_handles_error(mock_run: MagicMock) -> None:
    """Test getting NGC CLI version when command fails."""
    mock_run.side_effect = subprocess.CalledProcessError(1, "ngc")

    version = get_ngc_version()

    assert version is None


@patch("data_designer.cli.utils.subprocess.run")
def test_get_ngc_version_handles_timeout(mock_run: MagicMock) -> None:
    """Test getting NGC CLI version when command times out."""
    mock_run.side_effect = subprocess.TimeoutExpired("ngc", 5)

    version = get_ngc_version()

    assert version is None


@patch("data_designer.cli.utils.subprocess.run")
def test_get_ngc_version_handles_file_not_found(mock_run: MagicMock) -> None:
    """Test getting NGC CLI version when NGC CLI is not found."""
    mock_run.side_effect = FileNotFoundError()

    version = get_ngc_version()

    assert version is None
