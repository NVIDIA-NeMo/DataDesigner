# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.provider_controller import ProviderController
from data_designer.cli.repositories.model_repository import ModelConfigRegistry
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry
from data_designer.config.models import ModelConfig, ModelProvider


@pytest.fixture
def controller(tmp_path: Path) -> ProviderController:
    """Create a controller instance for testing."""
    return ProviderController(tmp_path)


@pytest.fixture
def controller_with_providers(
    controller: ProviderController, stub_model_providers: list[ModelProvider]
) -> ProviderController:
    """Create a controller instance with existing providers."""
    controller.repository.save(
        ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name)
    )
    return controller


@pytest.fixture
def controller_with_providers_and_models(
    controller_with_providers: ProviderController, stub_model_configs: list[ModelConfig]
) -> ProviderController:
    """Create a controller instance with existing providers and models."""
    controller_with_providers.model_repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    return controller_with_providers


def test_init(tmp_path: Path) -> None:
    """Test controller initialization sets up repositories and services correctly."""
    controller = ProviderController(tmp_path)
    assert controller.config_dir == tmp_path
    assert controller.repository.config_dir == tmp_path
    assert controller.service.repository == controller.repository
    assert controller.model_repository.config_dir == tmp_path
    assert controller.model_service.repository == controller.model_repository


def test_run_with_no_providers_and_user_cancels(controller: ProviderController) -> None:
    """Test run with no existing providers prompts for add and handles cancellation."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = None

    with patch("data_designer.cli.controllers.provider_controller.ProviderFormBuilder", return_value=mock_builder):
        controller.run()

    # Verify no providers were added since user cancelled
    assert len(controller.service.list_all()) == 0


@patch("data_designer.cli.controllers.provider_controller.select_with_arrows", return_value="no")
def test_run_with_no_providers_adds_new_provider(
    mock_select: MagicMock,
    controller: ProviderController,
    stub_new_model_provider: ModelProvider,
) -> None:
    """Test run with no existing providers successfully adds a new provider."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = stub_new_model_provider

    with patch("data_designer.cli.controllers.provider_controller.ProviderFormBuilder", return_value=mock_builder):
        controller.run()

    # Verify provider was actually added through the public interface
    providers = controller.service.list_all()
    assert len(providers) == 1
    assert providers[0].name == stub_new_model_provider.name
    assert providers[0].endpoint == stub_new_model_provider.endpoint


@patch("data_designer.cli.controllers.provider_controller.select_with_arrows", return_value="exit")
def test_run_with_existing_providers_and_exit(
    mock_select: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run with existing providers shows config and respects exit choice."""
    initial_count = len(controller_with_providers.service.list_all())

    controller_with_providers.run()

    # Verify no changes were made
    assert len(controller_with_providers.service.list_all()) == initial_count


@patch("data_designer.cli.controllers.provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.provider_controller.select_with_arrows")
def test_run_deletes_provider_without_models(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run can delete a provider through delete mode when no models are associated."""
    mock_select.side_effect = ["delete", "test-provider-1"]

    controller_with_providers.run()

    # Verify provider was actually deleted
    remaining_providers = controller_with_providers.service.list_all()
    assert len(remaining_providers) == 1
    assert remaining_providers[0].name == "test-provider-2"


@patch("data_designer.cli.controllers.provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.provider_controller.select_with_arrows")
def test_run_deletes_provider_with_associated_models(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers_and_models: ProviderController,
) -> None:
    """Test run deletes provider and associated models when confirmed."""
    mock_select.side_effect = ["delete", "test-provider-1"]

    controller_with_providers_and_models.run()

    # Verify provider and associated models were actually deleted
    providers = controller_with_providers_and_models.service.list_all()
    models = controller_with_providers_and_models.model_service.list_all()
    assert len(providers) == 1
    assert providers[0].name == "test-provider-2"
    assert len(models) == 0  # Both models were using test-provider-1


@patch("data_designer.cli.controllers.provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.provider_controller.select_with_arrows", return_value="delete_all")
def test_run_deletes_all_providers_without_models(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run can delete all providers through delete_all mode."""
    controller_with_providers.run()

    # Verify all providers were actually deleted
    assert len(controller_with_providers.service.list_all()) == 0


@patch("data_designer.cli.controllers.provider_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.provider_controller.select_with_arrows", return_value="delete_all")
def test_run_deletes_all_providers_with_models(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers_and_models: ProviderController,
) -> None:
    """Test run deletes all providers and associated models when confirmed."""
    controller_with_providers_and_models.run()

    # Verify all providers and models were actually deleted
    assert len(controller_with_providers_and_models.service.list_all()) == 0
    assert len(controller_with_providers_and_models.model_service.list_all()) == 0


@patch("data_designer.cli.controllers.provider_controller.select_with_arrows")
def test_run_updates_provider(
    mock_select: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run can update an existing provider through update mode."""
    mock_select.side_effect = ["update", "test-provider-1"]

    updated_provider = ModelProvider(
        name="test-provider-1-updated",
        endpoint="https://api.example.com/updated",
        provider_type="openai",
        api_key="updated-key",
    )

    mock_builder = MagicMock()
    mock_builder.run.return_value = updated_provider

    with patch("data_designer.cli.controllers.provider_controller.ProviderFormBuilder", return_value=mock_builder):
        controller_with_providers.run()

    # Verify provider was actually updated
    providers = controller_with_providers.service.list_all()
    assert len(providers) == 2
    updated = controller_with_providers.service.get_by_name("test-provider-1-updated")
    assert updated is not None
    assert updated.endpoint == "https://api.example.com/updated"
    assert controller_with_providers.service.get_by_name("test-provider-1") is None


@patch("data_designer.cli.controllers.provider_controller.select_with_arrows")
def test_run_changes_default_provider(
    mock_select: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run can change the default provider through change_default mode."""
    mock_select.side_effect = ["change_default", "test-provider-2"]

    # Verify initial default
    assert controller_with_providers.service.get_default() == "test-provider-1"

    controller_with_providers.run()

    # Verify default was actually changed
    assert controller_with_providers.service.get_default() == "test-provider-2"


@patch("data_designer.cli.controllers.provider_controller.confirm_action", return_value=False)
@patch("data_designer.cli.controllers.provider_controller.select_with_arrows")
def test_run_respects_delete_cancellation(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_providers: ProviderController,
) -> None:
    """Test run respects user's choice to cancel deletion."""
    mock_select.side_effect = ["delete", "test-provider-1"]

    initial_count = len(controller_with_providers.service.list_all())

    controller_with_providers.run()

    # Verify no providers were deleted
    assert len(controller_with_providers.service.list_all()) == initial_count
