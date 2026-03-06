# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from data_designer.config.models import (
    ChatCompletionInferenceParams,
    ModelConfig,
    ModelProvider,
)
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.factory import create_model_client
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.secret_resolver import SecretResolver


@pytest.fixture
def openai_registry() -> ModelProviderRegistry:
    provider = ModelProvider(name="openai-prod", endpoint="https://api.openai.com/v1", provider_type="openai")
    return ModelProviderRegistry(providers=[provider])


@pytest.fixture
def anthropic_registry() -> ModelProviderRegistry:
    provider = ModelProvider(name="anthropic-prod", endpoint="https://api.anthropic.com", provider_type="anthropic")
    return ModelProviderRegistry(providers=[provider])


@pytest.fixture
def openai_model_config() -> ModelConfig:
    return ModelConfig(
        alias="test-model",
        model="gpt-test",
        inference_parameters=ChatCompletionInferenceParams(),
        provider="openai-prod",
    )


@pytest.fixture
def anthropic_model_config() -> ModelConfig:
    return ModelConfig(
        alias="test-anthropic",
        model="claude-test",
        inference_parameters=ChatCompletionInferenceParams(),
        provider="anthropic-prod",
    )


@pytest.fixture
def secret_resolver() -> SecretResolver:
    resolver = MagicMock(spec=SecretResolver)
    resolver.resolve.return_value = "resolved-key"
    return resolver


# --- Provider routing ---


def test_openai_provider_creates_native_client(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(
        openai_model_config,
        secret_resolver,
        openai_registry,
        retry_config=RetryConfig(),
    )
    assert isinstance(client, OpenAICompatibleClient)


@patch("data_designer.engine.models.clients.factory.CustomRouter")
@patch("data_designer.engine.models.clients.factory.LiteLLMRouterDefaultKwargs")
def test_non_openai_provider_creates_bridge_client(
    mock_kwargs: MagicMock,
    mock_router: MagicMock,
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    anthropic_registry: ModelProviderRegistry,
) -> None:
    mock_kwargs.return_value.model_dump.return_value = {}
    client = create_model_client(anthropic_model_config, secret_resolver, anthropic_registry)
    assert isinstance(client, LiteLLMBridgeClient)


# --- Backend env var override ---


@patch("data_designer.engine.models.clients.factory.CustomRouter")
@patch("data_designer.engine.models.clients.factory.LiteLLMRouterDefaultKwargs")
def test_bridge_env_override_forces_bridge_for_openai_provider(
    mock_kwargs: MagicMock,
    mock_router: MagicMock,
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    mock_kwargs.return_value.model_dump.return_value = {}
    with patch.dict("os.environ", {"DATA_DESIGNER_MODEL_BACKEND": "litellm_bridge"}):
        client = create_model_client(openai_model_config, secret_resolver, openai_registry)
    assert isinstance(client, LiteLLMBridgeClient)


def test_native_env_var_still_uses_native_for_openai_provider(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    with patch.dict("os.environ", {"DATA_DESIGNER_MODEL_BACKEND": "native"}):
        client = create_model_client(
            openai_model_config,
            secret_resolver,
            openai_registry,
        )
    assert isinstance(client, OpenAICompatibleClient)
