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


@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_with_no_models_and_user_cancels(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    controller: ModelController,
) -> None:
    """Test run with no existing models prompts for add and handles cancellation."""
    mock_builder = MagicMock()
    mock_builder.run.return_value = None

    with patch("data_designer.cli.controllers.model_controller.ModelFormBuilder", return_value=mock_builder):
        controller.run()

    mock_print_header.assert_called_once_with("Configure Models")
    mock_print_info.assert_any_call("No models configured yet")
    # Verify no models were added since user cancelled
    assert len(controller.model_service.list_all()) == 0


@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="no")
@patch("data_designer.cli.controllers.model_controller.print_success")
@patch("data_designer.cli.controllers.model_controller.print_text")
@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_with_no_models_adds_new_model(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    mock_print_text: MagicMock,
    mock_print_success: MagicMock,
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
    mock_print_success.assert_called_once_with(f"Model '{stub_new_model_config.alias}' added successfully")


@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="exit")
@patch("data_designer.cli.controllers.model_controller.display_config_preview")
@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_with_existing_models_and_exit(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    mock_display: MagicMock,
    mock_select: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run with existing models shows config and respects exit choice."""
    initial_count = len(controller_with_models.model_service.list_all())

    controller_with_models.run()

    mock_print_header.assert_called_once_with("Configure Models")
    mock_display.assert_called_once()
    mock_print_info.assert_any_call("No changes made")
    # Verify no changes were made
    assert len(controller_with_models.model_service.list_all()) == initial_count


@patch("data_designer.cli.controllers.model_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.model_controller.select_with_arrows")
@patch("data_designer.cli.controllers.model_controller.print_success")
@patch("data_designer.cli.controllers.model_controller.display_config_preview")
@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_deletes_model(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    mock_display: MagicMock,
    mock_print_success: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can delete a model through delete mode."""
    # Setup: User selects delete mode, then selects first model
    mock_select.side_effect = ["delete", "test-alias-1"]

    initial_models = controller_with_models.model_service.list_all()
    assert len(initial_models) == 2

    controller_with_models.run()

    # Verify model was actually deleted
    remaining_models = controller_with_models.model_service.list_all()
    assert len(remaining_models) == 1
    assert remaining_models[0].alias == "test-alias-2"
    mock_print_success.assert_called_once_with("Model 'test-alias-1' deleted successfully")


@patch("data_designer.cli.controllers.model_controller.confirm_action", return_value=True)
@patch("data_designer.cli.controllers.model_controller.select_with_arrows", return_value="delete_all")
@patch("data_designer.cli.controllers.model_controller.print_success")
@patch("data_designer.cli.controllers.model_controller.display_config_preview")
@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_deletes_all_models(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    mock_display: MagicMock,
    mock_print_success: MagicMock,
    mock_select: MagicMock,
    mock_confirm: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can delete all models through delete_all mode."""
    assert len(controller_with_models.model_service.list_all()) == 2

    controller_with_models.run()

    # Verify all models were actually deleted
    assert len(controller_with_models.model_service.list_all()) == 0
    mock_print_success.assert_called_once_with("All (2) model(s) deleted successfully")


@patch("data_designer.cli.controllers.model_controller.select_with_arrows")
@patch("data_designer.cli.controllers.model_controller.print_success")
@patch("data_designer.cli.controllers.model_controller.display_config_preview")
@patch("data_designer.cli.controllers.model_controller.console.print")
@patch("data_designer.cli.controllers.model_controller.print_info")
@patch("data_designer.cli.controllers.model_controller.print_header")
def test_run_updates_model(
    mock_print_header: MagicMock,
    mock_print_info: MagicMock,
    mock_console_print: MagicMock,
    mock_display: MagicMock,
    mock_print_success: MagicMock,
    mock_select: MagicMock,
    controller_with_models: ModelController,
) -> None:
    """Test run can update an existing model through update mode."""
    # Setup: User selects update mode, then selects first model
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
    mock_print_success.assert_called_once_with("Model 'test-alias-1-updated' updated successfully")
