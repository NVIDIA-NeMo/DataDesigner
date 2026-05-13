# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from data_designer.config.column_configs import (
    CustomColumnConfig,
    ExpressionColumnConfig,
    GenerationStrategy,
    LLMTextColumnConfig,
)
from data_designer.config.custom_column import custom_column_generator
from data_designer.config.models import ChatCompletionInferenceParams, ModelConfig
from data_designer.engine.column_generators.generators.base import ColumnGenerator
from data_designer.engine.column_generators.generators.custom import CustomColumnGenerator
from data_designer.engine.dataset_builders.utils.scheduling_hints import SchedulingHint, SchedulingHintResolver
from data_designer.engine.resources.resource_provider import ResourceProvider

MODEL_ALIAS = "stub"


def _expr_config(name: str = "test") -> ExpressionColumnConfig:
    return ExpressionColumnConfig(name=name, expr="{{ x }}", dtype="str")


def _provider_with_model_configs(configs: dict[str, ModelConfig]) -> MagicMock:
    provider = MagicMock(spec=ResourceProvider)
    provider.model_registry = MagicMock()
    provider.model_registry.get_model_config.side_effect = lambda model_alias: configs[model_alias]
    provider.model_registry.get_model_provider.return_value = SimpleNamespace(name="mock-provider")
    return provider


class LocalCellGenerator(ColumnGenerator[ExpressionColumnConfig]):
    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = "local"
        return data


class ModelCellGenerator(ColumnGenerator[LLMTextColumnConfig]):
    @property
    def is_llm_bound(self) -> bool:
        return True

    @staticmethod
    def get_generation_strategy() -> GenerationStrategy:
        return GenerationStrategy.CELL_BY_CELL

    def generate(self, data: dict) -> dict:
        data[self.config.name] = "model"
        return data

    def get_model_config(self, model_alias: str) -> ModelConfig:
        return self.resource_provider.model_registry.get_model_config(model_alias=model_alias)

    def get_model_provider_name(self, model_alias: str) -> str:
        provider = self.resource_provider.model_registry.get_model_provider(model_alias=model_alias)
        return str(provider.name)


def test_scheduling_hint_resolver_local_hint_does_not_touch_model_registry() -> None:
    provider = MagicMock(spec=ResourceProvider)
    provider.model_registry = MagicMock()
    generator = LocalCellGenerator(config=_expr_config("local_col"), resource_provider=provider)

    resolver = SchedulingHintResolver({"local_col": generator})

    assert resolver.hint_for(generator) == SchedulingHint(group_kind="local")
    provider.model_registry.get_model_config.assert_not_called()
    provider.model_registry.get_model_provider.assert_not_called()


def test_scheduling_hint_resolver_resolves_primary_model_once_per_generator() -> None:
    model_config = ModelConfig(
        alias=MODEL_ALIAS,
        model="model-text",
        inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=3),
        provider="mock-provider",
    )
    provider = _provider_with_model_configs({MODEL_ALIAS: model_config})
    column_config = LLMTextColumnConfig(name="answer", prompt="hello", model_alias=MODEL_ALIAS)
    generator = ModelCellGenerator(config=column_config, resource_provider=provider)

    resolver = SchedulingHintResolver({"answer": generator, "answer_again": generator})
    hint = resolver.hint_for(generator)

    assert hint.group_kind == "model"
    assert hint.identity_prefix[:2] == ("mock-provider", "model-text")
    assert hint.weight == 3
    assert provider.model_registry.get_model_config.call_count == 1
    assert provider.model_registry.get_model_provider.call_count == 1


def test_scheduling_hint_resolver_falls_back_to_custom_model_hint_with_debug(
    caplog: pytest.LogCaptureFixture,
) -> None:
    provider = MagicMock(spec=ResourceProvider)
    provider.model_registry = MagicMock()
    provider.model_registry.get_model_config.side_effect = RuntimeError("registry unavailable")
    provider.model_registry.get_model_provider.return_value = SimpleNamespace(name="mock-provider")
    column_config = LLMTextColumnConfig(name="answer", prompt="hello", model_alias=MODEL_ALIAS)
    generator = ModelCellGenerator(config=column_config, resource_provider=provider)

    with caplog.at_level("DEBUG", logger="data_designer.engine.dataset_builders.utils.scheduling_hints"):
        resolver = SchedulingHintResolver({"answer": generator})

    hint = resolver.hint_for(generator)

    assert hint == SchedulingHint(group_kind="custom_model", identity_suffix=(MODEL_ALIAS,), weight=1)
    fallback_records = [
        record for record in caplog.records if "Falling back to custom-model scheduling group" in record.getMessage()
    ]
    assert len(fallback_records) == 1
    assert "answer" in fallback_records[0].getMessage()
    assert MODEL_ALIAS in fallback_records[0].getMessage()
    assert fallback_records[0].exc_info is not None


def test_scheduling_hint_resolver_partial_alias_fallback_preserves_resolved_weight() -> None:
    @custom_column_generator(model_aliases=["resolved", "missing"])
    def gen_with_models(row: dict, generator_params: None, models: dict) -> dict:
        row["custom_llm"] = "value"
        return row

    provider = _provider_with_model_configs(
        {
            "resolved": ModelConfig(
                alias="resolved",
                model="model-resolved",
                inference_parameters=ChatCompletionInferenceParams(max_parallel_requests=7),
                provider="mock-provider",
            )
        }
    )
    config = CustomColumnConfig(name="custom_llm", generator_function=gen_with_models)
    generator = CustomColumnGenerator(config=config, resource_provider=provider)

    resolver = SchedulingHintResolver({"custom_llm": generator})
    hint = resolver.hint_for(generator)

    assert hint == SchedulingHint(group_kind="custom_model", identity_suffix=("missing", "resolved"), weight=7)
