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
    from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
    from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController


def create_model_registry(
    *,
    model_configs: list[ModelConfig] | None = None,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
    mcp_registry: MCPRegistry | None = None,
    client_concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
    run_config: RunConfig | None = None,
    request_admission: AdaptiveRequestAdmissionController | None = None,
) -> ModelRegistry:
    """Factory function for creating a ModelRegistry instance.

    Heavy dependencies (httpx, etc.) are deferred until this function is called.
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
        run_config: Optional runtime configuration. Public request-admission
            tuning is translated to the engine-internal request-admission config.
        request_admission: Optional shared request-admission controller. When
            omitted, a new controller is created from ``run_config``.

    Returns:
        A configured ModelRegistry instance.
    """
    from data_designer.engine.models.clients.factory import create_model_client
    from data_designer.engine.models.clients.retry import RetryConfig
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.registry import ModelRegistry

    if request_admission is None:
        request_admission = create_request_admission_controller(run_config)

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
            request_admission=request_admission,
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
        request_admission=request_admission,
        retry_config=RetryConfig(),
    )


def create_request_admission_controller(
    run_config: RunConfig | None = None,
) -> AdaptiveRequestAdmissionController:
    """Create a request-admission controller from public runtime tuning."""
    from data_designer.config.run_config import RunConfig
    from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
    from data_designer.engine.models.request_admission.controller import AdaptiveRequestAdmissionController

    resolved_run_config = run_config or RunConfig()
    return AdaptiveRequestAdmissionController(
        _request_admission_config_from_run_config(resolved_run_config, RequestAdmissionConfig)
    )


def _request_admission_config_from_run_config(
    run_config: RunConfig,
    config_cls: type[RequestAdmissionConfig],
) -> RequestAdmissionConfig:
    tuning = run_config.request_admission
    if tuning is None:
        return config_cls()
    return config_cls(
        cooldown_seconds=tuning.cooldown_seconds,
        multiplicative_decrease_factor=tuning.multiplicative_decrease_factor,
        additive_increase_step=tuning.additive_increase_step,
        successes_until_increase=tuning.successes_until_increase,
        startup_ramp_seconds=tuning.startup_ramp_seconds,
    )
