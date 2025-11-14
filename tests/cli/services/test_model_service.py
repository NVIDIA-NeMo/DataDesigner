# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.cli.repositories.model_repository import ModelRepository
from data_designer.cli.services.model_service import ModelService
from data_designer.config.models import InferenceParameters, ModelConfig


def test_list_all(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    assert stub_model_service.list_all() == stub_model_configs


def test_get_by_alias(
    stub_model_service: ModelService,
    stub_model_configs: list[ModelConfig],
):
    assert stub_model_service.get_by_alias("test-alias-1") == stub_model_configs[0]
    assert stub_model_service.get_by_alias("test-alias-2") == stub_model_configs[1]
    assert stub_model_service.get_by_alias("test-alias-3") is None


def test_add(
    stub_model_service: ModelService, stub_model_configs: list[ModelConfig], stub_new_model_config: ModelConfig
):
    stub_model_service.add(stub_new_model_config)
    assert stub_model_service.list_all() == stub_model_configs + [stub_new_model_config]


def test_add_duplicate_alias(stub_model_service: ModelService, stub_inference_parameters: InferenceParameters):
    """Test adding a model with an alias that already exists."""
    duplicate_model = ModelConfig(
        alias="test-alias-1",
        model="test-model-duplicate",
        provider="test-provider-1",
        inference_parameters=stub_inference_parameters,
    )
    with pytest.raises(ValueError, match="Model alias 'test-alias-1' already exists"):
        stub_model_service.add(duplicate_model)


def test_update(stub_model_service: ModelService, stub_new_model_config: ModelConfig):
    stub_model_service.update("test-alias-1", stub_new_model_config)
    assert stub_model_service.get_by_alias("test-alias-1") is None
    assert stub_model_service.get_by_alias("test-alias-3") == stub_new_model_config


def test_update_no_registry(tmp_path: Path, stub_new_model_config: ModelConfig):
    """Test updating when no registry exists."""
    service = ModelService(ModelRepository(tmp_path))
    with pytest.raises(ValueError, match="No models configured"):
        service.update("test-alias-1", stub_new_model_config)


def test_update_nonexistent_model(stub_model_service: ModelService, stub_new_model_config: ModelConfig):
    """Test updating a model that doesn't exist."""
    with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
        stub_model_service.update("nonexistent", stub_new_model_config)


def test_update_to_existing_alias(stub_model_service: ModelService, stub_inference_parameters: InferenceParameters):
    """Test updating a model to an alias that already exists."""
    updated_model = ModelConfig(
        alias="test-alias-2",  # Already exists
        model="test-model-updated",
        provider="test-provider-1",
        inference_parameters=stub_inference_parameters,
    )
    with pytest.raises(ValueError, match="Model alias 'test-alias-2' already exists"):
        stub_model_service.update("test-alias-1", updated_model)


def test_delete(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    stub_model_service.delete("test-alias-1")
    assert stub_model_service.list_all() == stub_model_configs[1:]


def test_delete_no_registry(tmp_path: Path):
    """Test deleting when no registry exists."""
    service = ModelService(ModelRepository(tmp_path))
    with pytest.raises(ValueError, match="No models configured"):
        service.delete("test-alias-1")


def test_delete_nonexistent_model(stub_model_service: ModelService):
    """Test deleting a model that doesn't exist."""
    with pytest.raises(ValueError, match="Model 'nonexistent' not found"):
        stub_model_service.delete("nonexistent")


def test_delete_last_model(stub_model_service: ModelService):
    """Test deleting all models triggers repository.delete()."""
    stub_model_service.delete("test-alias-1")
    stub_model_service.delete("test-alias-2")
    assert stub_model_service.list_all() == []


def test_find_by_provider(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Both test models have provider="test-provider-1"
    models = stub_model_service.find_by_provider("test-provider-1")
    assert len(models) == 2
    assert models == stub_model_configs

    # Non-existent provider should return empty list
    models = stub_model_service.find_by_provider("non-existent-provider")
    assert models == []


def test_delete_by_aliases(stub_model_service: ModelService):
    # Delete both models
    stub_model_service.delete_by_aliases(["test-alias-1", "test-alias-2"])
    assert stub_model_service.list_all() == []


def test_delete_by_aliases_partial(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Delete only one model
    stub_model_service.delete_by_aliases(["test-alias-1"])
    assert stub_model_service.list_all() == stub_model_configs[1:]


def test_delete_by_aliases_empty_list(stub_model_service: ModelService, stub_model_configs: list[ModelConfig]):
    # Deleting empty list should do nothing
    stub_model_service.delete_by_aliases([])
    assert stub_model_service.list_all() == stub_model_configs


def test_delete_by_aliases_no_registry(tmp_path: Path):
    """Test delete_by_aliases when no registry exists."""
    service = ModelService(ModelRepository(tmp_path))
    with pytest.raises(ValueError, match="No models configured"):
        service.delete_by_aliases(["test-alias-1"])
