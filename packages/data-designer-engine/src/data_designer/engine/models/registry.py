# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from data_designer.config.models import GenerationType, ModelConfig
from data_designer.engine.model_provider import ModelProvider, ModelProviderRegistry
from data_designer.engine.models.usage import ModelUsageStats, RequestUsageStats, TokenUsageStats, ToolUsageStats
from data_designer.engine.secret_resolver import SecretResolver

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
            logger.info("  |-- no model usage recorded")
            return

        for model_name in sorted(model_usage_stats):
            stats = model_usage_stats[model_name]
            logger.info(f"  |-------- {model_name}")

            token_usage = stats.get("token_usage", {})
            input_tokens = int(token_usage.get("input_tokens", 0))
            output_tokens = int(token_usage.get("output_tokens", 0))
            total_tokens = int(token_usage.get("total_tokens", input_tokens + output_tokens))
            tokens_per_second = int(stats.get("tokens_per_second", 0))
            logger.info(
                f"  |-- tokens: input={input_tokens}, output={output_tokens}, total={total_tokens}, tps={tokens_per_second}"
            )

            request_usage = stats.get("request_usage", {})
            successful_requests = int(request_usage.get("successful_requests", 0))
            failed_requests = int(request_usage.get("failed_requests", 0))
            total_requests = int(request_usage.get("total_requests", successful_requests + failed_requests))
            requests_per_minute = int(stats.get("requests_per_minute", 0))
            logger.info(
                "  |-- requests: "
                f"success={successful_requests}, failed={failed_requests}, total={total_requests}, "
                f"rpm={requests_per_minute}"
            )

            tool_usage = stats.get("tool_usage")
            if tool_usage:
                total_tool_calls = int(tool_usage.get("total_tool_calls", 0))
                total_tool_call_turns = int(tool_usage.get("total_tool_call_turns", 0))
                generations_with_tools = int(tool_usage.get("generations_with_tools", 0))
                calls_mean = float(tool_usage.get("calls_per_generation_mean", 0.0))
                calls_stddev = float(tool_usage.get("calls_per_generation_stddev", 0.0))
                turns_mean = float(tool_usage.get("turns_per_generation_mean", 0.0))
                turns_stddev = float(tool_usage.get("turns_per_generation_stddev", 0.0))
                logger.info(
                    "  |-- tools: "
                    f"calls={total_tool_calls}, turns={total_tool_call_turns}, generations={generations_with_tools}, "
                    f"calls/gen={calls_mean:.1f} +/- {calls_stddev:.1f}, "
                    f"turns/gen={turns_mean:.1f} +/- {turns_stddev:.1f}"
                )

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

    def get_tool_usage_snapshot(self, *, model_alias: str) -> ToolUsageStats:
        return self.get_model(model_alias=model_alias).usage_stats.tool_usage.model_copy(deep=True)

    def get_tool_usage_delta(self, *, model_alias: str, snapshot: ToolUsageStats) -> ToolUsageStats:
        """Get the change in tool usage stats since a snapshot.

        Note: The returned delta object supports mean calculations but stddev values
        will be NaN since sum of squares cannot be computed from deltas alone.
        """
        current = self.get_model(model_alias=model_alias).usage_stats.tool_usage
        return ToolUsageStats(
            total_tool_calls=current.total_tool_calls - snapshot.total_tool_calls,
            total_tool_call_turns=current.total_tool_call_turns - snapshot.total_tool_call_turns,
            generations_with_tools=current.generations_with_tools - snapshot.generations_with_tools,
        )

    def get_model_provider(self, *, model_alias: str) -> ModelProvider:
        model_config = self.get_model_config(model_alias=model_alias)
        return self._model_provider_registry.get_provider(model_config.provider)

    def run_health_check(self, model_aliases: list[str]) -> None:
        logger.info("🩺 Running health checks for models...")
        for model_alias in model_aliases:
            model_config = self.get_model_config(model_alias=model_alias)
            if model_config.skip_health_check:
                logger.info(f"  |-- ⏭️  Skipping health check for model alias {model_alias!r} (skip_health_check=True)")
                continue

            model = self.get_model(model_alias=model_alias)
            logger.info(
                f"  |-- 👀 Checking {model.model_name!r} in provider named {model.model_provider_name!r} for model alias {model.model_alias!r}..."
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
                else:
                    raise ValueError(f"Unsupported generation type: {model.model_generation_type}")
                logger.info("  |-- ✅ Passed!")
            except Exception as e:
                logger.error("  |-- ❌ Failed!")
                raise e

    def _set_model_configs(self, model_configs: list[ModelConfig]) -> None:
        model_configs = model_configs or []
        self._model_configs = {mc.alias: mc for mc in model_configs}
        # Models are now lazily initialized in get_model() when first requested

    def _get_model(self, model_config: ModelConfig) -> ModelFacade:
        if self._model_facade_factory is None:
            raise RuntimeError("ModelRegistry was not initialized with a model_facade_factory")
        return self._model_facade_factory(model_config, self._secret_resolver, self._model_provider_registry)
