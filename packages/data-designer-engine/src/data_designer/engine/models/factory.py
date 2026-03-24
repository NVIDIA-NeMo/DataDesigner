# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.models.clients.adapters.http_model_client import ClientConcurrencyMode
from data_designer.engine.secret_resolver import SecretResolver

if TYPE_CHECKING:
    from data_designer.config.run_config import RunConfig
    from data_designer.engine.mcp.registry import MCPRegistry
    from data_designer.engine.models.registry import ModelRegistry


def create_model_registry(
    *,
    model_configs: list[ModelConfig] | None = None,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    mcp_registry: MCPRegistry | None = None,
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    run_config: RunConfig | None = None,
) -> ModelRegistry:
    """Factory function for creating a ModelRegistry instance.

    Heavy dependencies (litellm, httpx) are deferred until this function is called.
    This is a factory function pattern - imports inside factories are idiomatic Python
    for lazy initialization.

    Args:
        model_configs: Optional list of model configurations to register.
        secret_resolver: Resolver for secrets referenced in provider configs.
        model_provider_registry: Registry of model provider configurations.
        mcp_registry: Optional MCP registry for tool operations. When provided,
            ModelFacades can look up MCPFacades by tool_alias for tool-enabled generation.
        client_concurrency_mode: ``"sync"`` (default) or ``"async"``.  Forwarded
            to native HTTP adapters so each client is constrained to a single
            concurrency mode.
        run_config: Optional runtime configuration.  The nested
            ``run_config.throttle`` (a ``ThrottleConfig``) is forwarded to the
            ``ThrottleManager`` constructor.

    Returns:
        A configured ModelRegistry instance.
    """
    from data_designer.config.run_config import RunConfig
    from data_designer.engine.models.clients.factory import create_model_client
    from data_designer.engine.models.clients.retry import RetryConfig
    from data_designer.engine.models.clients.throttle_manager import ThrottleManager
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.litellm_overrides import apply_litellm_patches
    from data_designer.engine.models.registry import ModelRegistry

    apply_litellm_patches()

    throttle_manager = ThrottleManager((run_config or RunConfig()).throttle)

    def model_facade_factory(
        model_config: ModelConfig,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
        retry_config: RetryConfig | None,
    ) -> ModelFacade:
        client = create_model_client(
            model_config,
            secret_resolver,
            model_provider_registry,
            retry_config=retry_config,
            client_concurrency_mode=client_concurrency_mode,
            throttle_manager=throttle_manager,
        )
        return ModelFacade(
            model_config,
            model_provider_registry,
            client=client,
            mcp_registry=mcp_registry,
        )

    return ModelRegistry(
        model_configs=model_configs,
        secret_resolver=secret_resolver,
        model_provider_registry=model_provider_registry,
        model_facade_factory=model_facade_factory,
        throttle_manager=throttle_manager,
        retry_config=RetryConfig(),
    )
