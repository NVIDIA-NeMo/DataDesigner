# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.config.models import ModelConfig
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.anthropic import AnthropicClient
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.models.clients.adapters.openai_compatible import OpenAICompatibleClient
from data_designer.engine.models.clients.base import ModelClient
from data_designer.engine.models.clients.model_request_executor import ModelRequestExecutor
from data_designer.engine.models.clients.retry import RetryConfig
from data_designer.engine.models.errors import FormattedLLMErrorMessage
from data_designer.engine.models.request_admission.controller import RequestAdmissionController
from data_designer.engine.observability import RequestAdmissionEventSink
from data_designer.engine.secret_resolver import SecretResolver

_SUPPORTED_PROVIDER_TYPES = ("openai", "anthropic")
_NO_TRANSPORT_RETRY_CONFIG = RetryConfig(max_retries=0, retryable_status_codes=frozenset())


def create_model_client(
    model_config: ModelConfig,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    *,
    retry_config: RetryConfig | None = None,
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    request_admission: RequestAdmissionController | None = None,
    request_event_sink: RequestAdmissionEventSink | None = None,
) -> ModelClient:
    """Create a ``ModelClient`` for the given model configuration.

    Args:
        model_config: Model configuration specifying alias, model ID, provider,
            and inference parameters.
        secret_resolver: Resolver for secrets referenced in provider API key configs.
        model_provider_registry: Registry of model provider configurations used
            to look up endpoint, provider type, and API key reference.
        retry_config: Optional retry configuration for HTTP adapters.
        client_concurrency_mode: ``"sync"`` for synchronous adapter calls or
            ``"async"`` for async adapter calls. Native HTTP adapters are
            constrained to a single concurrency mode.
        request_admission: Optional request-admission controller for per-request
            provider/model/domain admission. When provided, the returned client
            is wrapped with ``ModelRequestExecutor``.

            **Ordering invariant:** the ``(provider_name, model_id)`` pair must
            be registered on the request-admission controller via ``register()`` before
            the returned client makes its first request.  In the standard flow,
            ``ModelRegistry._get_model()`` calls ``register()`` during model
            setup, which happens before any generation task invokes the client.
            Direct callers of this factory must ensure registration happens
            before use.

    Returns:
        A ``ModelClient`` instance routed by provider type.

    Raises:
        DataDesignerError: If ``provider_type`` is not one of the supported
            types (``"openai"``, ``"anthropic"``).

    Routing logic:
    1. If ``provider_type == "openai"`` → ``OpenAICompatibleClient``.
    2. If ``provider_type == "anthropic"`` → ``AnthropicClient``.
    3. Otherwise → ``DataDesignerError``.
    """
    provider = model_provider_registry.get_provider(model_config.provider)
    api_key = _resolve_api_key(provider.api_key, secret_resolver)
    max_parallel = model_config.inference_parameters.max_parallel_requests
    raw_timeout = model_config.inference_parameters.timeout
    timeout_s = float(raw_timeout if raw_timeout is not None else 60)
    adapter_retry_config = _NO_TRANSPORT_RETRY_CONFIG if request_admission is not None else retry_config

    if provider.provider_type == "openai":
        client: ModelClient = OpenAICompatibleClient(
            provider_name=provider.name,
            endpoint=provider.endpoint,
            api_key=api_key,
            retry_config=adapter_retry_config,
            max_parallel_requests=max_parallel,
            timeout_s=timeout_s,
            concurrency_mode=client_concurrency_mode,
        )
    elif provider.provider_type == "anthropic":
        client = AnthropicClient(
            provider_name=provider.name,
            endpoint=provider.endpoint,
            api_key=api_key,
            retry_config=adapter_retry_config,
            max_parallel_requests=max_parallel,
            timeout_s=timeout_s,
            concurrency_mode=client_concurrency_mode,
        )
    else:
        raise DataDesignerError(
            FormattedLLMErrorMessage(
                cause=(f"Provider type {provider.provider_type!r} for provider {provider.name!r} is not supported."),
                solution=(
                    f"Change provider_type to one of {', '.join(repr(t) for t in _SUPPORTED_PROVIDER_TYPES)} "
                    "in your model provider config. Most OpenAI-compatible endpoints "
                    '(vLLM, TGI, NIM, etc.) work with provider_type="openai".'
                ),
            )
        )

    if request_admission is not None:
        client = ModelRequestExecutor(
            inner=client,
            request_admission=request_admission,
            provider_name=provider.name,
            model_id=model_config.model,
            event_sink=request_event_sink,
            retry_config=retry_config,
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
