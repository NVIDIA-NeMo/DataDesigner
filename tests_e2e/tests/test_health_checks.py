# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from data_designer.config.models import (
    ChatCompletionInferenceParams,
    EmbeddingInferenceParams,
    ModelConfig,
    ModelProvider,
)
from data_designer.config.utils.constants import (
    NVIDIA_API_KEY_ENV_VAR_NAME,
    NVIDIA_PROVIDER_NAME,
    OPENAI_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME,
    OPENROUTER_API_KEY_ENV_VAR_NAME,
    OPENROUTER_PROVIDER_NAME,
    PREDEFINED_PROVIDERS,
    PREDEFINED_PROVIDERS_MODEL_MAP,
)
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.secret_resolver import EnvironmentResolver

PROVIDER_API_KEY_ENV_VARS = {
    NVIDIA_PROVIDER_NAME: NVIDIA_API_KEY_ENV_VAR_NAME,
    OPENAI_PROVIDER_NAME: OPENAI_API_KEY_ENV_VAR_NAME,
    OPENROUTER_PROVIDER_NAME: OPENROUTER_API_KEY_ENV_VAR_NAME,
}


def _get_provider_registry(provider_name: str) -> ModelProviderRegistry:
    """Create a registry with just the specified provider."""
    provider_data = next(p for p in PREDEFINED_PROVIDERS if p["name"] == provider_name)
    provider = ModelProvider(**provider_data)
    return ModelProviderRegistry(providers=[provider])


def _skip_if_no_api_key(provider_name: str) -> None:
    """Skip the test if the API key for the provider is not set."""
    env_var = PROVIDER_API_KEY_ENV_VARS[provider_name]
    if not os.environ.get(env_var):
        pytest.skip(f"{env_var} not set")


def _build_test_cases() -> list[tuple[str, str]]:
    """Build parametrized test cases: (provider_name, model_type)."""
    cases = []
    for provider_name in PROVIDER_API_KEY_ENV_VARS:
        for model_type in PREDEFINED_PROVIDERS_MODEL_MAP[provider_name]:
            cases.append((provider_name, model_type))
    return cases


@pytest.mark.parametrize("provider_name,model_type", _build_test_cases())
def test_health_check(provider_name: str, model_type: str) -> None:
    """Health check for each model in each default provider."""
    _skip_if_no_api_key(provider_name)

    provider_registry = _get_provider_registry(provider_name)
    secret_resolver = EnvironmentResolver()

    model_info = PREDEFINED_PROVIDERS_MODEL_MAP[provider_name][model_type]
    model_name = model_info["model"]
    inference_params = model_info["inference_parameters"]

    if model_type == "embedding":
        params = EmbeddingInferenceParams(**inference_params)
    else:
        params = ChatCompletionInferenceParams(**inference_params)

    model_config = ModelConfig(
        alias=f"{provider_name}-{model_type}",
        model=model_name,
        inference_parameters=params,
        provider=provider_name,
    )

    facade = ModelFacade(model_config, secret_resolver, provider_registry)

    if model_type == "embedding":
        result = facade.generate_text_embeddings(
            input_texts=["Hello!"],
            skip_usage_tracking=True,
        )
        assert len(result) == 1
        assert len(result[0]) > 0
    else:
        result, _ = facade.generate(
            prompt="Say 'OK' and nothing else.",
            parser=lambda x: x,
            system_prompt="You are a helpful assistant.",
            max_correction_steps=0,
            max_conversation_restarts=0,
            skip_usage_tracking=True,
        )
        assert isinstance(result, str)
        assert len(result) > 0
