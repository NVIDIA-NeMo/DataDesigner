# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProviderRegistry
from data_designer.engine.secret_resolver import SecretResolver

if TYPE_CHECKING:
    from data_designer.engine.models.registry import ModelRegistry


def create_model_registry(
    *,
    model_configs: list[ModelConfig] | None = None,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
) -> ModelRegistry:
    """Factory function for creating a ModelRegistry instance.

    Heavy dependencies (litellm, httpx) are deferred until this function is called.
    This is a factory function pattern - imports inside factories are idiomatic Python
    for lazy initialization.
    """
    from data_designer.engine.models.facade import ModelFacade
    from data_designer.engine.models.litellm_overrides import apply_litellm_patches
    from data_designer.engine.models.registry import ModelRegistry

    apply_litellm_patches()

    def model_facade_factory(model_config, secret_resolver, model_provider_registry):
        return ModelFacade(model_config, secret_resolver, model_provider_registry)

    return ModelRegistry(
        model_configs=model_configs,
        secret_resolver=secret_resolver,
        model_provider_registry=model_provider_registry,
        model_facade_factory=model_facade_factory,
    )
