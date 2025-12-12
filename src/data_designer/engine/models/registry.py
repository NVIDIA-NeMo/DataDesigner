# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

from data_designer.config.models import ModelConfig
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.facade import ModelFacade
from data_designer.engine.models.litellm_overrides import apply_litellm_patches
from data_designer.engine.secret_resolver import SecretResolver

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(
        self,
        *,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
        model_configs: list[ModelConfig] | None = None,
    ):
        self._secret_resolver = secret_resolver
        self._model_provider_registry = model_provider_registry
        self._model_configs = {}
        self._models: dict[str, ModelFacade] = {}
        self._set_model_configs(model_configs)

    @property
    def model_configs(self) -> dict[str, ModelConfig]:
        return self._model_configs

    @property
    def models(self) -> dict[str, ModelFacade]:
        return self._models

    def register_model_configs(self, model_configs: list[ModelConfig]) -> None:
        """Register a new Model configuration at runtime.

        Args:
            model_config: A new Model configuration to register. If an
                Model configuration already exists in the registry
                with the same name, then it will be overwritten.
        """
        self._set_model_configs(list(self._model_configs.values()) + model_configs)

    def get_model(self, *, model_alias: str) -> ModelFacade:
        # Check if model config exists first
        if model_alias not in self._model_configs:
            raise ValueError(f"No model config with alias {model_alias!r} found!")

        # Lazy initialization: only create model facade when first requested
        if model_alias not in self._models:
            self._models[model_alias] = self._get_model(self._model_configs[model_alias])

        return self._models[model_alias]

    def get_model_config(self, *, model_alias: str) -> ModelConfig:
        if model_alias not in self._model_configs:
            raise ValueError(f"No model config with alias {model_alias!r} found!")
        return self._model_configs[model_alias]

    def get_model_usage_stats(self, total_time_elapsed: float) -> dict[str, dict]:
        return {
            model.model_name: model.usage_stats.get_usage_stats(total_time_elapsed=total_time_elapsed)
            for model in self._models.values()
            if model.usage_stats.has_usage
        }

    def get_model_provider(self, *, model_alias: str) -> ModelProvider:
        model_config = self.get_model_config(model_alias=model_alias)
        return self._model_provider_registry.get_provider(model_config.provider)

    def run_health_check(self, model_aliases: set[str]) -> None:
        logger.info("🩺 Running health checks for models...")
        for model_alias in model_aliases:
            model = self.get_model(model_alias=model_alias)
            logger.info(
                f"  |-- 👀 Checking {model.model_name!r} in provider named {model.model_provider_name!r} for model alias {model.model_alias!r}..."
            )
            try:
                model.generate(
                    prompt="Hello!",
                    parser=lambda x: x,
                    system_prompt="You are a helpful assistant.",
                    max_correction_steps=0,
                    max_conversation_restarts=0,
                    skip_usage_tracking=True,
                    purpose="running health checks",
                )
                logger.info("  |-- ✅ Passed!")
            except Exception as e:
                logger.error("  |-- ❌ Failed!")
                raise e

    def _set_model_configs(self, model_configs: list[ModelConfig]) -> None:
        model_configs = model_configs or []
        self._model_configs = {mc.alias: mc for mc in model_configs}
        # Models are now lazily initialized in get_model() when first requested

    def _get_model(self, model_config: ModelConfig) -> ModelFacade:
        return ModelFacade(model_config, self._secret_resolver, self._model_provider_registry)


def create_model_registry(
    *,
    model_configs: list[ModelConfig] | None = None,
    secret_resolver: SecretResolver,
    model_provider_registry: ModelProviderRegistry,
) -> ModelRegistry:
    apply_litellm_patches()
    return ModelRegistry(
        model_configs=model_configs,
        secret_resolver=secret_resolver,
        model_provider_registry=model_provider_registry,
    )
