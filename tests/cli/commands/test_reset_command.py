# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import typer

from data_designer.cli.commands.reset import reset_command
from data_designer.config.utils.constants import DATA_DESIGNER_HOME

# Type alias for the factory function
MockRepositoryFactory = Callable[
    [bool, bool, Exception | None, Exception | None],
    tuple[Mock, Mock, Mock, Mock],
]


# Fixtures for common test data
@pytest.fixture
def stub_fake_provider_path() -> Path:
    """Fake path for provider config file."""
    return Path("/fake/providers.json")


@pytest.fixture
def stub_fake_model_path() -> Path:
    """Fake path for model config file."""
    return Path("/fake/models.json")


# Helper functions for mock setup
def setup_mock_repository(
    exists: bool = True,
    config_file: Path | None = None,
    delete_side_effect: Exception | None = None,
) -> Mock:
    """Create a mock repository instance with common configuration.

    Args:
        exists: Whether the config file exists
        config_file: Path to the config file
        delete_side_effect: Optional exception to raise on delete()
    """
    mock_instance = Mock()
    mock_instance.exists.return_value = exists
    if config_file:
        mock_instance.config_file = config_file
    if delete_side_effect:
        mock_instance.delete.side_effect = delete_side_effect
    return mock_instance


@pytest.fixture
def mock_repositories_factory(stub_fake_provider_path: Path, stub_fake_model_path: Path) -> MockRepositoryFactory:
    """Factory fixture for creating mock repositories with different configurations."""

    def _factory(
        provider_exists: bool = False,
        model_exists: bool = False,
        provider_delete_error: Exception | None = None,
        model_delete_error: Exception | None = None,
    ) -> tuple[Mock, Mock, Mock, Mock]:
        """Create mocked repositories and their instances.

        Returns:
            Tuple of (mock_provider_repo, mock_provider_instance,
                     mock_model_repo, mock_model_instance)
        """
        mock_provider_instance = setup_mock_repository(
            exists=provider_exists,
            config_file=stub_fake_provider_path if provider_exists else None,
            delete_side_effect=provider_delete_error,
        )
        mock_provider_repo = Mock(return_value=mock_provider_instance)

        mock_model_instance = setup_mock_repository(
            exists=model_exists,
            config_file=stub_fake_model_path if model_exists else None,
            delete_side_effect=model_delete_error,
        )
        mock_model_repo = Mock(return_value=mock_model_instance)

        return mock_provider_repo, mock_provider_instance, mock_model_repo, mock_model_instance

    return _factory


# Tests
@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_no_config_files_exist(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
) -> None:
    """Test reset when no configuration files exist - should exit early."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=False, model_exists=False
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance

    with pytest.raises(typer.Exit) as exc_info:
        reset_command()

    assert exc_info.value.exit_code == 0
    mock_confirm.assert_not_called()
    mock_provider_instance.delete.assert_not_called()
    mock_model_instance.delete.assert_not_called()


@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_both_files_exist_user_confirms_both(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
) -> None:
    """Test reset when both config files exist and user confirms deletion of both."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=True, model_exists=True
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance
    mock_confirm.return_value = True

    reset_command()

    assert mock_confirm.call_count == 2
    mock_provider_instance.delete.assert_called_once()
    mock_model_instance.delete.assert_called_once()


@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_both_files_exist_user_declines_both(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
) -> None:
    """Test reset when both config files exist but user declines deletion."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=True, model_exists=True
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance
    mock_confirm.return_value = False

    reset_command()

    assert mock_confirm.call_count == 2
    mock_provider_instance.delete.assert_not_called()
    mock_model_instance.delete.assert_not_called()


@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_mixed_confirmation(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
) -> None:
    """Test reset when user confirms one file but not the other."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=True, model_exists=True
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance
    mock_confirm.side_effect = [True, False]

    reset_command()

    assert mock_confirm.call_count == 2
    mock_provider_instance.delete.assert_called_once()
    mock_model_instance.delete.assert_not_called()


@pytest.mark.parametrize(
    "provider_error,model_error,expected_provider_calls,expected_model_calls",
    [
        (Exception("Permission denied"), None, 1, 1),
        (None, OSError("Disk error"), 1, 1),
        (Exception("Error 1"), Exception("Error 2"), 1, 1),
    ],
    ids=["provider_fails", "model_fails", "both_fail"],
)
@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_deletion_failures(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
    provider_error: Exception | None,
    model_error: Exception | None,
    expected_provider_calls: int,
    expected_model_calls: int,
) -> None:
    """Test reset when deletion fails for one or more repositories."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=True,
        model_exists=True,
        provider_delete_error=provider_error,
        model_delete_error=model_error,
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance
    mock_confirm.return_value = True

    with pytest.raises(typer.Exit) as exc_info:
        reset_command()

    assert exc_info.value.exit_code == 1
    assert mock_provider_instance.delete.call_count == expected_provider_calls
    assert mock_model_instance.delete.call_count == expected_model_calls


@pytest.mark.parametrize(
    "provider_exists,model_exists,expected_confirms,expected_provider_deletes,expected_model_deletes",
    [
        (True, False, 1, 1, 0),
        (False, True, 1, 0, 1),
    ],
    ids=["only_provider", "only_model"],
)
@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_single_file_exists(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
    provider_exists: bool,
    model_exists: bool,
    expected_confirms: int,
    expected_provider_deletes: int,
    expected_model_deletes: int,
) -> None:
    """Test reset when only one config file exists."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=provider_exists, model_exists=model_exists
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance
    mock_confirm.return_value = True

    reset_command()

    assert mock_confirm.call_count == expected_confirms
    assert mock_provider_instance.delete.call_count == expected_provider_deletes
    assert mock_model_instance.delete.call_count == expected_model_deletes


@patch("data_designer.cli.commands.reset.ModelRepository")
@patch("data_designer.cli.commands.reset.ProviderRepository")
@patch("data_designer.cli.commands.reset.confirm_action")
def test_reset_uses_default_config_dir_when_none_provided(
    mock_confirm: Mock,
    mock_provider_repo: Mock,
    mock_model_repo: Mock,
    mock_repositories_factory: MockRepositoryFactory,
) -> None:
    """Test that default config directory is used when config_dir is None."""
    _, mock_provider_instance, _, mock_model_instance = mock_repositories_factory(
        provider_exists=False, model_exists=False
    )
    mock_provider_repo.return_value = mock_provider_instance
    mock_model_repo.return_value = mock_model_instance

    with pytest.raises(typer.Exit):
        reset_command()

    mock_provider_repo.assert_called_once_with(DATA_DESIGNER_HOME)
    mock_model_repo.assert_called_once_with(DATA_DESIGNER_HOME)
