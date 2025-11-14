# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.cli.repositories.model_repository import ModelConfigRegistry, ModelRepository
from data_designer.cli.repositories.provider_repository import ModelProviderRegistry, ProviderRepository
from data_designer.cli.services.model_service import ModelService
from data_designer.cli.services.provider_service import ProviderService
from data_designer.config.models import InferenceParameters, ModelConfig, ModelProvider


@pytest.fixture
def stub_inference_parameters() -> InferenceParameters:
    return InferenceParameters(temperature=0.7, top_p=0.9, max_tokens=2048, max_parallel_requests=4)


@pytest.fixture
def stub_model_configs(stub_inference_parameters: InferenceParameters) -> list[ModelConfig]:
    return [
        ModelConfig(
            alias="test-alias-1",
            model="test-model-1",
            provider="test-provider-1",
            inference_parameters=stub_inference_parameters,
        ),
        ModelConfig(
            alias="test-alias-2",
            model="test-model-2",
            provider="test-provider-1",
            inference_parameters=stub_inference_parameters,
        ),
    ]


@pytest.fixture
def stub_new_model_config() -> ModelConfig:
    return ModelConfig(
        alias="test-alias-3",
        model="test-model-3",
        provider="test-provider-1",
        inference_parameters=InferenceParameters(
            temperature=0.7,
            top_p=0.9,
            max_tokens=2048,
            max_parallel_requests=4,
            timeout=100,
        ),
    )


@pytest.fixture
def stub_model_providers() -> list[ModelProvider]:
    return [
        ModelProvider(
            name="test-provider-1",
            endpoint="https://api.example.com/v1",
            provider_type="openai",
            api_key="test-api-key",
        ),
        ModelProvider(
            name="test-provider-2",
            endpoint="https://api.example.com/v2",
            provider_type="openai",
            api_key="test-api-key-2",
        ),
    ]


@pytest.fixture
def stub_new_model_provider() -> ModelProvider:
    return ModelProvider(
        name="test-provider-3",
        endpoint="https://api.example.com/v1",
        provider_type="openai",
        api_key="test-api-key-1",
    )


@pytest.fixture
def stub_model_service(tmp_path: Path, stub_model_configs: list[ModelConfig]) -> ModelService:
    repository = ModelRepository(tmp_path)
    repository.save(ModelConfigRegistry(model_configs=stub_model_configs))
    return ModelService(repository)


@pytest.fixture
def stub_provider_service(tmp_path: Path, stub_model_providers: list[ModelProvider]) -> ProviderService:
    repository = ProviderRepository(tmp_path)
    repository.save(ModelProviderRegistry(providers=stub_model_providers, default=stub_model_providers[0].name))
    return ProviderService(repository)
