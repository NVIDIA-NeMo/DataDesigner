# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace
from typing import Literal
from unittest.mock import MagicMock

import pytest

from data_designer.config.base import SingleColumnConfig
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.models import GenerationType
from data_designer.config.scheduling import SchedulingMetadata, SchedulingMetadataError
from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
from data_designer.engine.dataset_builders.scheduling.resolver import TaskSchedulingResolver
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey


class _LocalGenerator:
    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return SchedulingMetadata.local()


class _ModelGenerator:
    def __init__(self, metadata: SchedulingMetadata) -> None:
        self._metadata = metadata

    def get_scheduling_metadata(self) -> SchedulingMetadata:
        return self._metadata


class _FallbackGenerator:
    def get_scheduling_metadata(self) -> SchedulingMetadata:
        raise SchedulingMetadataError(
            code="partial",
            message="using fallback",
            fallback=SchedulingMetadata.local("fallback"),
            diagnostics={"reason": "test"},
        )


class _FatalGenerator:
    def get_scheduling_metadata(self) -> SchedulingMetadata:
        raise SchedulingMetadataError(code="fatal", message="fatal")


def _task(column: str = "answer") -> Task:
    return Task(column=column, row_group=0, row_index=0, task_type="cell")


def test_task_scheduling_resolver_uses_local_default_metadata() -> None:
    resolver = TaskSchedulingResolver({"answer": _LocalGenerator()})  # type: ignore[arg-type]

    schedulable = resolver.schedulable_task(_task(), ("answer",))

    assert schedulable.group.key.kind == "local"
    assert schedulable.resource_request.amounts == {"submission": 1}


def test_task_scheduling_resolver_maps_model_metadata_to_model_resource() -> None:
    metadata = SchedulingMetadata.model("nvidia", "nemotron", "chat", weight=3)
    resolver = TaskSchedulingResolver({"answer": _ModelGenerator(metadata)})  # type: ignore[arg-type]

    schedulable = resolver.schedulable_task(_task(), ("answer",))

    assert schedulable.group.key.kind == "model"
    assert schedulable.group.weight == 3.0
    assert schedulable.group.admitted_limit == 6
    assert schedulable.resource_request.amounts == {"submission": 1, "llm_wait": 1}
    assert schedulable.request_resource_key == RequestResourceKey("nvidia", "nemotron", RequestDomain.CHAT)


def test_task_scheduling_resolver_records_safe_fallback_diagnostics() -> None:
    resolver = TaskSchedulingResolver({"answer": _FallbackGenerator()})  # type: ignore[arg-type]

    schedulable = resolver.schedulable_task(_task(), ("answer",))

    assert schedulable.group.key.identity[:2] == ("local", "fallback")
    assert resolver.diagnostics[0]["code"] == "partial"


def test_task_scheduling_resolver_raises_fatal_metadata_error() -> None:
    with pytest.raises(SchedulingMetadataError):
        TaskSchedulingResolver({"answer": _FatalGenerator()})  # type: ignore[arg-type]


def test_model_registry_generator_metadata_deduplicates_same_endpoint_aliases() -> None:
    class _RegistryGenerator(ColumnGeneratorWithModelRegistry[ExpressionColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> object:
            return object()

        def generate(self, data: object) -> object:
            return data

    config = ExpressionColumnConfig(name="answer", expr="{{ x }}", dtype="str")
    generator = _RegistryGenerator(config=config, resource_provider=MagicMock())
    generator._get_scheduling_model_aliases = lambda: ["primary", "secondary"]  # type: ignore[method-assign]
    configs = {
        "primary": SimpleNamespace(
            model="endpoint",
            generation_type=GenerationType.CHAT_COMPLETION,
            inference_parameters=SimpleNamespace(max_parallel_requests=4),
        ),
        "secondary": SimpleNamespace(
            model="endpoint",
            generation_type=GenerationType.CHAT_COMPLETION,
            inference_parameters=SimpleNamespace(max_parallel_requests=2),
        ),
    }
    providers = {
        "primary": SimpleNamespace(name="nvidia"),
        "secondary": SimpleNamespace(name="nvidia"),
    }
    generator.get_model_config = lambda model_alias: configs[model_alias]  # type: ignore[method-assign]
    generator.get_model_provider_name = lambda model_alias: providers[model_alias].name  # type: ignore[method-assign]

    metadata = generator.get_scheduling_metadata()

    assert metadata.identity == ("model", "nvidia", "endpoint", "chat")
    assert metadata.weight == 2
    assert metadata.diagnostics["merge_rule"] == "min_same_endpoint"

    resolver = TaskSchedulingResolver({"answer": generator})  # type: ignore[arg-type]
    schedulable = resolver.schedulable_task(_task(), ("answer",))
    assert schedulable.request_resource_key == RequestResourceKey("nvidia", "endpoint", RequestDomain.CHAT)


def test_model_registry_generator_metadata_uses_custom_model_for_multi_endpoint_aliases() -> None:
    class _PairwiseJudgeColumnConfig(SingleColumnConfig):
        column_type: Literal["pairwise-judge-test"] = "pairwise-judge-test"
        model_alias: str
        judge_model_alias: str

        @property
        def required_columns(self) -> list[str]:
            return []

        @property
        def side_effect_columns(self) -> list[str]:
            return []

        def get_model_aliases(self) -> list[str]:
            return [self.model_alias, self.judge_model_alias]

    class _RegistryGenerator(ColumnGeneratorWithModelRegistry[_PairwiseJudgeColumnConfig]):
        @staticmethod
        def get_generation_strategy() -> object:
            return object()

        def generate(self, data: object) -> object:
            return data

    config = _PairwiseJudgeColumnConfig(name="answer", model_alias="draft", judge_model_alias="judge")
    assert config.get_model_aliases() == ["draft", "judge"]
    generator = _RegistryGenerator(config=config, resource_provider=MagicMock())
    configs = {
        "draft": SimpleNamespace(
            model="draft-endpoint",
            generation_type=GenerationType.CHAT_COMPLETION,
            inference_parameters=SimpleNamespace(max_parallel_requests=4),
        ),
        "judge": SimpleNamespace(
            model="judge-endpoint",
            generation_type=GenerationType.CHAT_COMPLETION,
            inference_parameters=SimpleNamespace(max_parallel_requests=2),
        ),
    }
    providers = {
        "draft": SimpleNamespace(name="nvidia"),
        "judge": SimpleNamespace(name="openai"),
    }
    generator.get_model_config = lambda model_alias: configs[model_alias]  # type: ignore[method-assign]
    generator.get_model_provider_name = lambda model_alias: providers[model_alias].name  # type: ignore[method-assign]

    metadata = generator.get_scheduling_metadata()

    assert metadata.kind == "custom_model"
    assert metadata.identity[1].endswith("._RegistryGenerator")
    assert metadata.identity[2].startswith("alias-set-")
    assert metadata.weight == 6
    assert metadata.diagnostics["aliases"] == ("draft", "judge")
    assert metadata.diagnostics["fallback_reason"] == "multi_endpoint_alias_set"
    assert metadata.diagnostics["raw_caps"] == (4, 2)

    resolver = TaskSchedulingResolver({"answer": generator})  # type: ignore[arg-type]
    schedulable = resolver.schedulable_task(_task(), ("answer",))
    assert schedulable.group.key.kind == "custom_model"
    assert schedulable.request_resource_key is None
