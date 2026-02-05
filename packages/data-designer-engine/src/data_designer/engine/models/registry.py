# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from data_designer.config.models import GenerationType, ModelConfig
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats
from data_designer.engine.secret_resolver import SecretResolver
from data_designer.logging import LOG_INDENT

if TYPE_CHECKING:
    from data_designer.engine.models.facade import ModelFacade

logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(
        self,
        *,
        secret_resolver: SecretResolver,
        model_provider_registry: ModelProviderRegistry,
        model_configs: list[ModelConfig] | None = None,
        model_facade_factory: Callable[[ModelConfig, SecretResolver, ModelProviderRegistry], ModelFacade] | None = None,
    ):
        self._secret_resolver = secret_resolver
        self._model_provider_registry = model_provider_registry
        self._model_facade_factory = model_facade_factory
        self._model_configs: dict[str, ModelConfig] = {}
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

    def log_model_usage(self, total_time_elapsed: float) -> None:
        """Log a formatted summary of model usage statistics."""
        model_usage_stats = self.get_model_usage_stats(total_time_elapsed)

        logger.info("📊 Model usage summary:")
        if not model_usage_stats:
            logger.info(f"{LOG_INDENT}no model usage recorded")
            return

        sorted_model_names = sorted(model_usage_stats)
        for model_index, model_name in enumerate(sorted_model_names):
            stats = model_usage_stats[model_name]
            logger.info(f"{LOG_INDENT}model: {model_name}")

            token_usage = stats["token_usage"]
            input_tokens = token_usage["input_tokens"]
            output_tokens = token_usage["output_tokens"]
            total_tokens = token_usage["total_tokens"]
            tokens_per_second = stats["tokens_per_second"]
            logger.info(
                f"{LOG_INDENT}tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}, tps={tokens_per_second}"
            )

            request_usage = stats["request_usage"]
            successful_requests = request_usage["successful_requests"]
            failed_requests = request_usage["failed_requests"]
            total_requests = request_usage["total_requests"]
            requests_per_minute = stats["requests_per_minute"]
            logger.info(
                f"{LOG_INDENT}requests: "
                f"success={successful_requests}, failed={failed_requests}, total={total_requests}, "
                f"rpm={requests_per_minute}"
            )

            if tool_usage := stats.get("tool_usage"):
                total_gens = tool_usage["total_generations"]
                gens_with_tools = tool_usage["generations_with_tools"]
                logger.info(
                    f"{LOG_INDENT}tools: "
                    f"generations={gens_with_tools}/{total_gens}, "
                    f"calls={tool_usage['total_tool_calls']}, "
                    f"turns={tool_usage['total_tool_call_turns']}"
                )

            if model_index < len(sorted_model_names) - 1:
                logger.info(LOG_INDENT.rstrip())

    def get_model_usage_snapshot(self) -> dict[str, ModelUsageStats]:
        return {
            model.model_name: model.usage_stats.model_copy(deep=True)
            for model in self._models.values()
            if model.usage_stats.has_usage
        }

    def get_usage_deltas(self, snapshot: dict[str, ModelUsageStats]) -> dict[str, ModelUsageStats]:
        deltas = {}
        for model_name, current in self.get_model_usage_snapshot().items():
            prev = snapshot.get(model_name)
            delta_input = current.token_usage.input_tokens - (prev.token_usage.input_tokens if prev else 0)
            delta_output = current.token_usage.output_tokens - (prev.token_usage.output_tokens if prev else 0)
            delta_successful = current.request_usage.successful_requests - (
                prev.request_usage.successful_requests if prev else 0
            )
            delta_failed = current.request_usage.failed_requests - (prev.request_usage.failed_requests if prev else 0)

            if delta_input > 0 or delta_output > 0 or delta_successful > 0 or delta_failed > 0:
                deltas[model_name] = ModelUsageStats(
                    token_usage=TokenUsageStats(input_tokens=delta_input, output_tokens=delta_output),
                    request_usage=RequestUsageStats(successful_requests=delta_successful, failed_requests=delta_failed),
                )
        return deltas

    def get_model_provider(self, *, model_alias: str) -> ModelProvider:
        model_config = self.get_model_config(model_alias=model_alias)
        return self._model_provider_registry.get_provider(model_config.provider)

    def run_health_check(self, model_aliases: list[str]) -> None:
        logger.info("🩺 Running health checks for models...")
        for model_alias in model_aliases:
            model_config = self.get_model_config(model_alias=model_alias)
            if model_config.skip_health_check:
                logger.info(
                    f"{LOG_INDENT}⏭️  Skipping health check for model alias {model_alias!r} (skip_health_check=True)"
                )
                continue

            model = self.get_model(model_alias=model_alias)
            logger.info(
                f"{LOG_INDENT}👀 Checking {model.model_name!r} in provider named {model.model_provider_name!r} for model alias {model.model_alias!r}..."
            )
            try:
                if model.model_generation_type == GenerationType.EMBEDDING:
                    model.generate_text_embeddings(
                        input_texts=["Hello!"],
                        skip_usage_tracking=True,
                        purpose="running health checks",
                    )
                elif model.model_generation_type == GenerationType.CHAT_COMPLETION:
                    model.generate(
                        prompt="Hello!",
                        parser=lambda x: x,
                        system_prompt="You are a helpful assistant.",
                        max_correction_steps=0,
                        max_conversation_restarts=0,
                        skip_usage_tracking=True,
                        purpose="running health checks",
                    )
                elif model.model_generation_type == GenerationType.IMAGE_GENERATION:
                    model.generate_image(
                        prompt="Generate a simple pixel",
                        skip_usage_tracking=True,
                        purpose="running health checks",
                    )
                else:
                    raise ValueError(f"Unsupported generation type: {model.model_generation_type}")
                logger.info(f"{LOG_INDENT}✅ Passed!")
            except Exception as e:
                logger.error(f"{LOG_INDENT}❌ Failed!")
                raise e

    def _set_model_configs(self, model_configs: list[ModelConfig]) -> None:
        model_configs = model_configs or []
        self._model_configs = {mc.alias: mc for mc in model_configs}
        # Models are now lazily initialized in get_model() when first requested

    def _get_model(self, model_config: ModelConfig) -> ModelFacade:
        if self._model_facade_factory is None:
            raise RuntimeError("ModelRegistry was not initialized with a model_facade_factory")
        return self._model_facade_factory(model_config, self._secret_resolver, self._model_provider_registry)
