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
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.factory import create_model_client
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.throttle_manager import ThrottleManager
from data_designer.engine.models.clients.throttled import ThrottledModelClient
from data_designer.engine.secret_resolver import SecretResolver


@pytest.fixture
def openai_registry() -> ModelProviderRegistry:
    provider = ModelProvider(
        name="openai-prod", endpoint="https://api.openai.com/v1", provider_type="openai", api_key="env:OPENAI_KEY"
    )
    return ModelProviderRegistry(providers=[provider])


@pytest.fixture
def anthropic_registry() -> ModelProviderRegistry:
    provider = ModelProvider(
        name="anthropic-prod", endpoint="https://api.anthropic.com/v1", provider_type="anthropic", api_key="env:ANT_KEY"
    )
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
    secret_resolver.resolve.assert_called_once_with("env:OPENAI_KEY")


def test_anthropic_provider_creates_native_client(
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    anthropic_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(
        anthropic_model_config,
        secret_resolver,
        anthropic_registry,
        retry_config=RetryConfig(),
    )
    assert isinstance(client, AnthropicClient)
    secret_resolver.resolve.assert_called_once_with("env:ANT_KEY")


def test_anthropic_provider_type_case_insensitive(
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
) -> None:
    for variant in ("Anthropic", "ANTHROPIC", "anthropic"):
        provider = ModelProvider(
            name="anthropic-prod", endpoint="https://api.anthropic.com/v1", provider_type=variant, api_key="env:ANT_KEY"
        )
        registry = ModelProviderRegistry(providers=[provider])
        client = create_model_client(anthropic_model_config, secret_resolver, registry, retry_config=RetryConfig())
        assert isinstance(client, AnthropicClient), f"Failed for provider_type={variant!r}"


@patch("data_designer.engine.models.clients.factory.CustomRouter")
@patch("data_designer.engine.models.clients.factory.LiteLLMRouterDefaultKwargs")
def test_unknown_provider_creates_bridge_client(
    mock_kwargs: MagicMock,
    mock_router: MagicMock,
    secret_resolver: SecretResolver,
) -> None:
    mock_kwargs.return_value.model_dump.return_value = {}
    provider = ModelProvider(name="custom-provider", endpoint="https://custom.example.com", provider_type="custom")
    registry = ModelProviderRegistry(providers=[provider])
    config = ModelConfig(
        alias="test-custom",
        model="custom-model",
        inference_parameters=ChatCompletionInferenceParams(),
        provider="custom-provider",
    )
    client = create_model_client(config, secret_resolver, registry)
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


@patch("data_designer.engine.models.clients.factory.CustomRouter")
@patch("data_designer.engine.models.clients.factory.LiteLLMRouterDefaultKwargs")
def test_bridge_env_override_forces_bridge_for_anthropic_provider(
    mock_kwargs: MagicMock,
    mock_router: MagicMock,
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    anthropic_registry: ModelProviderRegistry,
) -> None:
    mock_kwargs.return_value.model_dump.return_value = {}
    with patch.dict("os.environ", {"DATA_DESIGNER_MODEL_BACKEND": "litellm_bridge"}):
        client = create_model_client(anthropic_model_config, secret_resolver, anthropic_registry)
    assert isinstance(client, LiteLLMBridgeClient)


def test_openai_provider_type_case_insensitive(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
) -> None:
    for variant in ("OpenAI", "OPENAI", "Openai"):
        provider = ModelProvider(
            name="openai-prod", endpoint="https://api.openai.com/v1", provider_type=variant, api_key="env:OPENAI_KEY"
        )
        registry = ModelProviderRegistry(providers=[provider])
        client = create_model_client(openai_model_config, secret_resolver, registry)
        assert isinstance(client, OpenAICompatibleClient), f"Failed for provider_type={variant!r}"


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


# --- Mode parameter forwarding ---


def test_concurrency_mode_forwarded_to_openai_client(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(
        openai_model_config, secret_resolver, openai_registry, client_concurrency_mode=ClientConcurrencyMode.ASYNC
    )
    assert isinstance(client, OpenAICompatibleClient)
    assert client.concurrency_mode == ClientConcurrencyMode.ASYNC


def test_concurrency_mode_forwarded_to_anthropic_client(
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    anthropic_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(
        anthropic_model_config, secret_resolver, anthropic_registry, client_concurrency_mode=ClientConcurrencyMode.ASYNC
    )
    assert isinstance(client, AnthropicClient)
    assert client.concurrency_mode == ClientConcurrencyMode.ASYNC


def test_concurrency_mode_defaults_to_sync(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(openai_model_config, secret_resolver, openai_registry)
    assert isinstance(client, OpenAICompatibleClient)
    assert client.concurrency_mode == ClientConcurrencyMode.SYNC


# --- Throttle manager wrapping ---


def test_throttle_manager_wraps_openai_client(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    tm = ThrottleManager()
    client = create_model_client(
        openai_model_config, secret_resolver, openai_registry, retry_config=RetryConfig(), throttle_manager=tm
    )
    assert isinstance(client, ThrottledModelClient)
    assert isinstance(client._inner, OpenAICompatibleClient)


def test_throttle_manager_wraps_anthropic_client(
    anthropic_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    anthropic_registry: ModelProviderRegistry,
) -> None:
    tm = ThrottleManager()
    client = create_model_client(
        anthropic_model_config, secret_resolver, anthropic_registry, retry_config=RetryConfig(), throttle_manager=tm
    )
    assert isinstance(client, ThrottledModelClient)
    assert isinstance(client._inner, AnthropicClient)


@patch("data_designer.engine.models.clients.factory.CustomRouter")
@patch("data_designer.engine.models.clients.factory.LiteLLMRouterDefaultKwargs")
def test_bridge_client_is_wrapped_with_throttle_manager(
    mock_kwargs: MagicMock,
    mock_router: MagicMock,
    secret_resolver: SecretResolver,
) -> None:
    """LiteLLMBridgeClient is wrapped, but AIMD accuracy is best-effort
    because the bridge's internal router may retry 429s before the
    wrapper sees them. See architecture notes for scope."""
    mock_kwargs.return_value.model_dump.return_value = {}
    provider = ModelProvider(name="custom-provider", endpoint="https://custom.example.com", provider_type="custom")
    registry = ModelProviderRegistry(providers=[provider])
    config = ModelConfig(
        alias="test-custom",
        model="custom-model",
        inference_parameters=ChatCompletionInferenceParams(),
        provider="custom-provider",
    )
    tm = ThrottleManager()
    client = create_model_client(config, secret_resolver, registry, throttle_manager=tm)
    assert isinstance(client, ThrottledModelClient)
    assert isinstance(client._inner, LiteLLMBridgeClient)


def test_no_throttle_manager_returns_inner_client_directly(
    openai_model_config: ModelConfig,
    secret_resolver: SecretResolver,
    openai_registry: ModelProviderRegistry,
) -> None:
    client = create_model_client(openai_model_config, secret_resolver, openai_registry)
    assert isinstance(client, OpenAICompatibleClient)
    assert not isinstance(client, ThrottledModelClient)
