# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from data_designer.cli.controllers.model_controller import ModelController
from data_designer.cli.repositories.model_repository import ModelConfigRegistry
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.config.models import InferenceParameters, ModelConfig


@pytest.fixture
def controller(tmp_path: Path, stub_model_providers: list) -> ModelController:
    """Create a controller instance for testing."""
    provider_repo = ProviderRepository(tmp_path)
    provider_repo.save(ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name))
    return ModelController(tmp_path)


@pytest.fixture
def controller_with_models(controller: ModelController, stub_model_configs: list[ModelConfig]) -> ModelController:
    """Create a controller instance with existing models."""
    controller.model_repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    return controller


def test_init(tmp_path: Path) -> None:
    """Test controller initialization sets up repositories and services correctly."""
    controller = ModelController(tmp_path)
    assert controller.config_dir == tmp_path
    assert controller.model_repository.config_dir == tmp_path
    assert controller.model_service.repository == controller.model_repository
    assert controller.provider_repository.config_dir == tmp_path
    assert controller.provider_service.repository == controller.provider_repository


@patch("data_designer.cli.controllers.model_controller.print_error")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_with_no_providers(
    mock_print_header: MagicMock, mock_print_info: MagicMock, mock_print_error: MagicMock, tmp_path: Path
) -> None:
    """Test run exits early when no providers are configured."""
    controller = ModelController(tmp_path)
    controller.run()

    mock_print_header.assert_called_once_with("Configure Models")
    mock_print_error.assert_called_once_with("No providers available!")
    mock_print_info.assert_called_once_with("Please run 'data-designer config providers' first")


def test_run_with_no_models_and_user_cancels(controller: ModelController) -> None:
    """Test run with no existing models prompts for add and handles cancellation."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = None

    with patch("data_designer.cli.controllers.model_controller.ModelFormBuilder", return_value=mock_builder):
        controller.run()

    # Verify no models were added since user cancelled
    assert len(controller.model_service.list_all()) == 0


@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="no")
def test_run_with_no_models_adds_new_model(
    mock_select: MagicMock,
    controller: ModelController,
    stub_new_model_config: ModelConfig,
) -> None:
    """Test run with no existing models successfully adds a new model."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = stub_new_model_config

    with patch("data_designer.cli.controllers.model_controller.ModelFormBuilder", return_value=mock_builder):
        controller.run()

    # Verify model was actually added through the public interface
    models = controller.model_service.list_all()
    assert len(models) == 1
    assert models[0].alias == stub_new_model_config.alias


@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="exit")
def test_run_with_existing_models_and_exit(
    mock_select: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run with existing models shows config and respects exit choice."""
    initial_count = len(controller_with_models.model_service.list_all())

    controller_with_models.run()

    # Verify no changes were made
    assert len(controller_with_models.model_service.list_all()) == initial_count


@patch("data_designer.cli.controllers.model_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.model_controller.select_with_arrows")
def test_run_deletes_model(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can delete a model through delete mode."""
    mock_select.side_effect = ["delete", "test-alias-1"]

    controller_with_models.run()

    # Verify model was actually deleted
    remaining_models = controller_with_models.model_service.list_all()
    assert len(remaining_models) == 1
    assert remaining_models[0].alias == "test-alias-2"


@patch("data_designer.cli.controllers.model_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="delete_all")
def test_run_deletes_all_models(
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can delete all models through delete_all mode."""
    controller_with_models.run()

    # Verify all models were actually deleted
    assert len(controller_with_models.model_service.list_all()) == 0


@patch("data_designer.cli.controllers.model_controller.select_with_arrows")
def test_run_updates_model(
    mock_select: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can update an existing model through update mode."""
    mock_select.side_effect = ["update", "test-alias-1"]

    updated_config = ModelConfig(
        alias="test-alias-1-updated",
        model="test-model-1-updated",
        provider="test-provider-1",
        inference_parameters=InferenceParameters(temperature=0.8, top_p=0.95, max_tokens=1024),
    )

    mock_builder = MagicMock()
    mock_builder.run.return_value = updated_config

    with patch("data_designer.cli.controllers.model_controller.ModelFormBuilder", return_value=mock_builder):
        controller_with_models.run()

    # Verify model was actually updated
    models = controller_with_models.model_service.list_all()
    assert len(models) == 2
    updated_model = controller_with_models.model_service.get_by_alias("test-alias-1-updated")
    assert updated_model is not None
    assert updated_model.model == "test-model-1-updated"
    assert controller_with_models.model_service.get_by_alias("test-alias-1") is None
