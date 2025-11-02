# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest
from typer.testing import CliRunner

from data_designer.cli import app
from data_designer.cli.utils import save_config_file

runner = CliRunner()


@pytest.fixture
def valid_provider_config(tmp_path: Path) -> Path:
    """Create a valid provider configuration file."""
    config = {
        "providers": [
            {
                "name": "test_provider",
                "endpoint": "https://test.api.com/v1",
                "provider_type": "openai",
                "api_key": "TEST_API_KEY",
            }
        ],
        "default": "test_provider",
    }
    provider_file = tmp_path / "model_providers.yaml"
    save_config_file(provider_file, config)
    return tmp_path


@pytest.fixture
def valid_model_config(tmp_path: Path) -> Path:
    """Create a valid model configuration file."""
    config = {
        "model_configs": [
            {
                "alias": "test_model",
                "model": "test/model-id",
                "inference_parameters": {
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2048,
                    "max_parallel_requests": 4,
                },
            }
        ]
    }
    model_file = tmp_path / "model_configs.yaml"
    save_config_file(model_file, config)
    return tmp_path


@pytest.fixture
def invalid_provider_config(tmp_path: Path) -> Path:
    """Create an invalid provider configuration file."""
    config = {
        "providers": [],  # Empty providers list is invalid
        "default": "nonexistent",
    }
    provider_file = tmp_path / "model_providers.yaml"
    save_config_file(provider_file, config)
    return tmp_path


@pytest.fixture
def invalid_model_config(tmp_path: Path) -> Path:
    """Create an invalid model configuration file."""
    config = {
        "model_configs": [
            {
                "alias": "test_model",
                "model": "test/model-id",
                "inference_parameters": {
                    "temperature": 5.0,  # Invalid - too high
                    "top_p": 0.9,
                    "max_tokens": 2048,
                },
            }
        ]
    }
    model_file = tmp_path / "model_configs.yaml"
    save_config_file(model_file, config)
    return tmp_path


def test_list_command_with_valid_configs(valid_provider_config: Path, valid_model_config: Path) -> None:
    """Test list command with valid configurations."""
    # Note: These fixtures point to the same tmp_path
    config_dir = valid_provider_config

    result = runner.invoke(app, ["config", "list", "--config-dir", str(config_dir)])

    assert result.exit_code == 0
    assert "Model Providers" in result.output
    assert "Model Configurations" in result.output
    assert "test_provider" in result.output
    assert "test_model" in result.output


def test_list_command_with_json_output(valid_provider_config: Path, valid_model_config: Path) -> None:
    """Test list command with JSON output."""
    config_dir = valid_provider_config

    result = runner.invoke(app, ["config", "list", "--config-dir", str(config_dir), "--json"])

    assert result.exit_code == 0
    assert '"providers"' in result.output
    assert '"models"' in result.output
    assert '"test_provider"' in result.output


def test_list_command_missing_configs(tmp_path: Path) -> None:
    """Test list command with missing configuration files."""
    result = runner.invoke(app, ["config", "list", "--config-dir", str(tmp_path)])

    assert result.exit_code == 0  # Should not error, just warn
    assert "not found" in result.output.lower()


def test_validate_command_valid_configs(valid_provider_config: Path, valid_model_config: Path) -> None:
    """Test validate command with valid configurations."""
    config_dir = valid_provider_config

    result = runner.invoke(app, ["config", "validate", "--config-dir", str(config_dir)])

    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_validate_command_invalid_provider(invalid_provider_config: Path) -> None:
    """Test validate command with invalid provider configuration."""
    config_dir = invalid_provider_config

    result = runner.invoke(app, ["config", "validate", "--config-dir", str(config_dir)])

    assert result.exit_code == 1
    assert "invalid" in result.output.lower() or "error" in result.output.lower()


def test_validate_command_invalid_model(valid_provider_config: Path) -> None:
    """Test validate command with invalid model configuration."""
    config_dir = valid_provider_config

    # Create an invalid model config directly
    invalid_config = {
        "model_configs": [
            {
                "alias": "test_model",
                "model": "test/model-id",
                "inference_parameters": {
                    "temperature": 5.0,  # Invalid - too high
                    "top_p": 0.9,
                    "max_tokens": 2048,
                },
            }
        ]
    }
    save_config_file(config_dir / "model_configs.yaml", invalid_config)

    result = runner.invoke(app, ["config", "validate", "--config-dir", str(config_dir)])

    assert result.exit_code == 1
    assert "invalid" in result.output.lower() or "error" in result.output.lower()


def test_validate_command_missing_configs(tmp_path: Path) -> None:
    """Test validate command with missing configuration files."""
    result = runner.invoke(app, ["config", "validate", "--config-dir", str(tmp_path)])

    assert result.exit_code == 1  # Should error
    assert "not found" in result.output.lower()


def test_validate_command_with_warnings(tmp_path: Path) -> None:
    """Test validate command with warnings (e.g., missing API keys)."""
    config = {
        "providers": [
            {
                "name": "test_provider",
                "endpoint": "https://test.api.com/v1",
                "provider_type": "openai",
                # No api_key - should generate warning
            }
        ],
        "default": "test_provider",
    }
    provider_file = tmp_path / "model_providers.yaml"
    save_config_file(provider_file, config)

    model_config = {
        "model_configs": [
            {
                "alias": "test_model",
                "model": "test/model-id",
                "inference_parameters": {
                    "temperature": 0.05,  # Very low - should generate warning
                    "top_p": 0.9,
                    "max_tokens": 2048,
                    "max_parallel_requests": 4,
                },
            }
        ]
    }
    model_file = tmp_path / "model_configs.yaml"
    save_config_file(model_file, model_config)

    result = runner.invoke(app, ["config", "validate", "--config-dir", str(tmp_path)])

    assert result.exit_code == 0  # Valid but with warnings
    assert "warning" in result.output.lower() or "⚠" in result.output
