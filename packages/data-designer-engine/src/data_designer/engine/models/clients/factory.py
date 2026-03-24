# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.litellm_bridge import LiteLLMBridgeClient
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.clients.throttle_manager import ThrottleManager
from data_designer.engine.models.clients.throttled import ThrottledModelClient
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
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    throttle_manager: ThrottleManager | None = None,
) -> ModelClient:
    """Create a ``ModelClient`` for the given model configuration.

    Args:
        model_config: Model configuration specifying alias, model ID, provider,
            and inference parameters.
        secret_resolver: Resolver for secrets referenced in provider API key configs.
        model_provider_registry: Registry of model provider configurations used
            to look up endpoint, provider type, and API key reference.
        retry_config: Optional retry configuration for native HTTP adapters.
            Ignored by the ``LiteLLMBridgeClient`` (which has its own retry logic).
        client_concurrency_mode: ``"sync"`` (default) for the sync engine path,
            ``"async"`` for the async engine path.  Native HTTP adapters are
            constrained to a single concurrency mode; the ``LiteLLMBridgeClient``
            ignores this parameter.
        throttle_manager: Optional throttle manager for per-request AIMD
            concurrency control.  When provided, the returned client is wrapped
            with ``ThrottledModelClient``.

            **Ordering invariant:** the ``(provider_name, model_id)`` pair must
            be registered on the ``ThrottleManager`` via ``register()`` before
            the returned client makes its first request.  In the standard flow,
            ``ModelRegistry._get_model()`` calls ``register()`` during model
            setup, which happens before any generation task invokes the client.
            Direct callers of this factory must ensure registration happens
            before use.

    Returns:
        A ``ModelClient`` instance routed by provider type.

    Routing logic:
    1. If ``DATA_DESIGNER_MODEL_BACKEND=litellm_bridge`` → always use bridge.
    2. If ``provider_type == "openai"`` → ``OpenAICompatibleClient``.
    3. If ``provider_type == "anthropic"`` → ``AnthropicClient``.
    4. Otherwise → ``LiteLLMBridgeClient`` (fallback for unknown providers).
    """
    provider = model_provider_registry.get_provider(model_config.provider)
    api_key = _resolve_api_key(provider.api_key, secret_resolver)
    max_parallel = model_config.inference_parameters.max_parallel_requests
    raw_timeout = model_config.inference_parameters.timeout
    timeout_s = float(raw_timeout if raw_timeout is not None else 60)

    backend = os.environ.get(_BACKEND_ENV_VAR, "").strip().lower()
    if backend == _BACKEND_BRIDGE:
        client: ModelClient = _create_bridge_client(model_config, provider, api_key, max_parallel)
    elif provider.provider_type == "openai":
        client = OpenAICompatibleClient(
            provider_name=provider.name,
            endpoint=provider.endpoint,
            api_key=api_key,
            retry_config=retry_config,
            max_parallel_requests=max_parallel,
            timeout_s=timeout_s,
            concurrency_mode=client_concurrency_mode,
        )
    elif provider.provider_type == "anthropic":
        client = AnthropicClient(
            provider_name=provider.name,
            endpoint=provider.endpoint,
            api_key=api_key,
            retry_config=retry_config,
            max_parallel_requests=max_parallel,
            timeout_s=timeout_s,
            concurrency_mode=client_concurrency_mode,
        )
    else:
        client = _create_bridge_client(model_config, provider, api_key, max_parallel)

    if throttle_manager is not None:
        client = ThrottledModelClient(
            inner=client,
            throttle_manager=throttle_manager,
            provider_name=provider.name,
            model_id=model_config.model,
        )

    return client


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
