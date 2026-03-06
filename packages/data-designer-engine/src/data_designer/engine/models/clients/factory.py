# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.litellm_overrides import CustomRouter, LiteLLMRouterDefaultKwargs
from data_designer.engine.secret_resolver import SecretResolver


def create_model_client(
    model_config: ModelConfig,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
) -> ModelClient:
    """Create a ModelClient for the given model configuration.

    Resolves the provider, API key, and constructs a LiteLLM router wrapped in
    a LiteLLMBridgeClient adapter.

    Args:
        model_config: The model configuration to create a client for.
        secret_resolver: Resolver for secrets referenced in provider configs.
        model_provider_registry: Registry of model provider configurations.

    Returns:
        A ModelClient instance ready for use.
    """
    provider = model_provider_registry.get_provider(model_config.provider)
    api_key = None
    if provider.api_key:
        api_key = secret_resolver.resolve(provider.api_key)
    api_key = api_key or "not-used-but-required"

    litellm_params = lazy.litellm.LiteLLM_Params(
        model=f"{provider.provider_type}/{model_config.model}",
        api_base=provider.endpoint,
        api_key=api_key,
        max_parallel_requests=model_config.inference_parameters.max_parallel_requests,
    )
    deployment = {
        "model_name": model_config.model,
        "litellm_params": litellm_params.model_dump(),
    }
    router = CustomRouter([deployment], **LiteLLMRouterDefaultKwargs().model_dump())
    return LiteLLMBridgeClient(provider_name=provider.name, router=router)
