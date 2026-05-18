# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from data_designer.config.scheduling import SchedulingMetadata, SchedulingMetadataError
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_model import Task

if TYPE_CHECKING:
    from data_designer.engine.column_generators.generators.base import ColumnGenerator


@dataclass(frozen=True)
class ResolvedTaskScheduling:
    """Scheduler inputs resolved from generator-facing metadata."""

    group: TaskGroupSpec
    resource_request: SchedulerResourceRequest


class TaskSchedulingResolver:
    """Resolve generator metadata into scheduler-internal task inputs."""

    def __init__(
        self,
        generators: Mapping[str, ColumnGenerator],
        *,
        model_group_limit_multiplier: int = 2,
        model_group_limit_cap: int = 256,
    ) -> None:
        self._generators = generators
        self._model_group_limit_multiplier = model_group_limit_multiplier
        self._model_group_limit_cap = model_group_limit_cap
        self._metadata_by_generator_id: dict[int, SchedulingMetadata] = {}
        self._diagnostics: list[dict[str, object]] = []
        for generator in dict.fromkeys(generators.values()):
            self._metadata_by_generator_id[id(generator)] = self._resolve_metadata(generator)

    @property
    def diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(self._diagnostics)

    def scheduling_for_task(self, task: Task, flow_identity: tuple[str, ...]) -> ResolvedTaskScheduling:
        generator = self._generators[task.column]
        metadata = self._metadata_by_generator_id[id(generator)]
        return self._resolved_from_metadata(metadata, flow_identity)

    def schedulable_task(self, task: Task, flow_identity: tuple[str, ...]) -> SchedulableTask:
        resolved = self.scheduling_for_task(task, flow_identity)
        return SchedulableTask(
            task_id=stable_task_id(task),
            payload=task,
            group=resolved.group,
            resource_request=resolved.resource_request,
        )

    def _resolve_metadata(self, generator: ColumnGenerator) -> SchedulingMetadata:
        try:
            return generator.get_scheduling_metadata()
        except SchedulingMetadataError as exc:
            if exc.fallback is None:
                raise
            self._diagnostics.append(
                {
                    "code": exc.code,
                    "message": exc.message,
                    "fallback": exc.fallback.identity,
                    "diagnostics": exc.diagnostics,
                }
            )
            return exc.fallback

    def _resolved_from_metadata(
        self,
        metadata: SchedulingMetadata,
        flow_identity: tuple[str, ...],
    ) -> ResolvedTaskScheduling:
        weight = max(1, metadata.weight)
        if metadata.kind == "local":
            key = TaskGroupKey(kind="local", identity=(*metadata.identity, *flow_identity))
            return ResolvedTaskScheduling(
                group=TaskGroupSpec(key=key, weight=float(weight)),
                resource_request=SchedulerResourceRequest({"submission": 1}),
            )

        identity = (*metadata.identity, *flow_identity)
        admitted_limit = max(1, min(self._model_group_limit_cap, self._model_group_limit_multiplier * weight))
        return ResolvedTaskScheduling(
            group=TaskGroupSpec(
                key=TaskGroupKey(kind=metadata.kind, identity=identity),
                weight=float(weight),
                admitted_limit=admitted_limit,
            ),
            resource_request=SchedulerResourceRequest({"submission": 1, "llm_wait": 1}),
        )
