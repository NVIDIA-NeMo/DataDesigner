# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pytest

from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    ModelConfig,
)
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.registry import ModelRegistry, create_model_registry
from data_designer.engine.secret_resolver import SecretsFileResolver


@pytest.fixture
def stub_secrets_resolver() -> SecretsFileResolver:
    module_path = Path(__file__).parent
    return SecretsFileResolver(module_path / "stub_secrets.json")


@pytest.fixture
def stub_model_provider_registry() -> ModelProviderRegistry:
    return ModelProviderRegistry(
        providers=[
            ModelProvider(
                name="stub-model-provider",
                endpoint="https://api.example.com/v1",
                provider_type="openai",
                api_key="STUB_API_KEY",
            )
        ]
    )


@pytest.fixture
def stub_model_configs() -> list[ModelConfig]:
    return [
        ModelConfig(
            alias="stub-text",
            model="stub-model-text",
            provider="stub-model-provider",
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.80, top_p=0.95, max_tokens=100, max_parallel_requests=10, timeout=100
            ),
        ),
        ModelConfig(
            alias="stub-reasoning",
            model="stub-model-reasoning",
            provider="stub-model-provider",
            inference_parameters=ChatCompletionInferenceParams(
                temperature=0.80, top_p=0.95, max_tokens=100, max_parallel_requests=10, timeout=100
            ),
        ),
        ModelConfig(
            alias="stub-embedding",
            model="stub-model-embedding",
            provider="stub-model-provider",
            inference_parameters=EmbeddingInferenceParams(
                dimensions=100,
            ),
        ),
    ]


@pytest.fixture
def stub_model_registry(stub_model_configs, stub_secrets_resolver, stub_model_provider_registry) -> ModelRegistry:
    return create_model_registry(
        model_configs=stub_model_configs,
        secret_resolver=stub_secrets_resolver,
        model_provider_registry=stub_model_provider_registry,
    )
