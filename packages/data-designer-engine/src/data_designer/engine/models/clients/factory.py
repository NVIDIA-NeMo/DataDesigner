# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.litellm_overrides import CustomRouter, LiteLLMRouterDefaultKwargs
from data_designer.engine.secret_resolver import SecretResolver

_BACKEND_ENV_VAR = "DATA_DESIGNER_MODEL_BACKEND"
_BACKEND_BRIDGE = "litellm_bridge"


def create_model_client(
    model_config: ModelConfig,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    *,
    retry_config: RetryConfig | None = None,
) -> ModelClient:
    """Create a ``ModelClient`` for the given model configuration.

    Routing logic:
    1. If ``DATA_DESIGNER_MODEL_BACKEND=litellm_bridge`` → always use bridge.
    2. If ``provider_type == "openai"`` → ``OpenAICompatibleClient``.
    3. Otherwise → ``LiteLLMBridgeClient`` (Anthropic native adapter is PR-4).
    """
    provider = model_provider_registry.get_provider(model_config.provider)
    api_key = _resolve_api_key(provider.api_key, secret_resolver)
    max_parallel = model_config.inference_parameters.max_parallel_requests
    raw_timeout = model_config.inference_parameters.timeout
    timeout_s = float(raw_timeout if raw_timeout is not None else 60)

    backend = os.environ.get(_BACKEND_ENV_VAR, "").strip().lower()
    use_native = backend != _BACKEND_BRIDGE and provider.provider_type == "openai"

    if use_native:
        return OpenAICompatibleClient(
            provider_name=provider.name,
            model_id=model_config.model,
            endpoint=provider.endpoint,
            api_key=api_key,
            retry_config=retry_config,
            max_parallel_requests=max_parallel,
            timeout_s=timeout_s,
        )

    return _create_bridge_client(model_config, provider, api_key, max_parallel)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _resolve_api_key(api_key_ref: str | None, secret_resolver: SecretResolver) -> str | None:
    if not api_key_ref:
        return None
    resolved = secret_resolver.resolve(api_key_ref)
    return resolved or None


def _create_bridge_client(
    model_config: ModelConfig,
    provider: ModelProvider,
    api_key: str | None,
    max_parallel: int,
) -> LiteLLMBridgeClient:
    bridge_key = api_key or "not-used-but-required"
    litellm_params = lazy.litellm.LiteLLM_Params(
        model=f"{provider.provider_type}/{model_config.model}",
        api_base=provider.endpoint,
        api_key=bridge_key,
        max_parallel_requests=max_parallel,
    )
    deployment = {
        "model_name": model_config.model,
        "litellm_params": litellm_params.model_dump(),
    }
    router = CustomRouter([deployment], **LiteLLMRouterDefaultKwargs().model_dump())
    return LiteLLMBridgeClient(provider_name=provider.name, router=router)
