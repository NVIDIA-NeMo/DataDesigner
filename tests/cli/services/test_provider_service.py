# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.services.model_service import ModelService
from data_designer.cli.services.provider_service import ProviderService
from data_designer.config.models import ModelConfig, ModelProvider


def test_list_all(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.list_all() == stub_model_providers


def test_get_by_name(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.get_by_name("test-provider-1") == stub_model_providers[0]
    assert stub_provider_service.get_by_name("test-provider-3") is None


def test_add(
    stub_provider_service: ProviderService,
    stub_model_providers: list[ModelProvider],
    stub_new_model_provider: ModelProvider,
):
    stub_provider_service.add(stub_new_model_provider)
    assert stub_provider_service.list_all() == stub_model_providers + [stub_new_model_provider]


def test_update(stub_provider_service: ProviderService, stub_new_model_provider: ModelProvider):
    stub_provider_service.update("test-provider-1", stub_new_model_provider)
    assert stub_provider_service.get_by_name("test-provider-1") is None
    assert stub_provider_service.get_by_name("test-provider-3") == stub_new_model_provider


def test_delete(stub_provider_service: ProviderService):
    stub_provider_service.delete("test-provider-1")
    assert len(stub_provider_service.list_all()) == 1


def test_set_default(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    stub_provider_service.set_default("test-provider-2")
    assert stub_provider_service.get_default() == "test-provider-2"


def test_get_default(stub_provider_service: ProviderService, stub_model_providers: list[ModelProvider]):
    assert stub_provider_service.get_default() == "test-provider-1"


def test_delete_provider_with_associated_models(
    tmp_path: Path, stub_model_providers: list[ModelProvider], stub_model_configs: list[ModelConfig]
):
    """Test integration: deleting a provider and its associated models."""
    # Setup: Create provider and model services
    provider_service = ProviderService(
        ModelRepository(tmp_path)  # This should be ProviderRepository
    )
    model_service = ModelService(ModelRepository(tmp_path))

    # Save provider and models
    from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository

    provider_repo = ProviderRepository(tmp_path)
    provider_repo.save(ModelProviderRegistry(providers=stub_model_providers, default="test-provider-1"))

    model_repo = ModelRepository(tmp_path)
    model_repo.save(ModelConfigRegistry(model_configs=stub_model_configs))

    # Verify initial state: 2 providers, 2 models (both associated with test-provider-1)
    provider_service = ProviderService(provider_repo)
    assert len(provider_service.list_all()) == 2
    assert len(model_service.list_all()) == 2

    # Find models associated with test-provider-1
    associated_models = model_service.find_by_provider("test-provider-1")
    assert len(associated_models) == 2

    # Delete the associated models
    model_aliases = [m.alias for m in associated_models]
    model_service.delete_by_aliases(model_aliases)

    # Delete the provider
    provider_service.delete("test-provider-1")

    # Verify final state: 1 provider, 0 models
    assert len(provider_service.list_all()) == 1
    assert len(model_service.list_all()) == 0
    assert provider_service.get_by_name("test-provider-1") is None
