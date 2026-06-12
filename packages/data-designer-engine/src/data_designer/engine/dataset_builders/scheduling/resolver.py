# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING

from data_designer.config.scheduling import SchedulingMetadata, SchedulingMetadataError
from data_designer.engine.dataset_builders.scheduling.resources import (
    SchedulableTask,
    SchedulerResourceKey,
    SchedulerResourceRequest,
    TaskGroupKey,
    TaskGroupSpec,
    request_scheduler_resource_key,
    stable_task_id,
)
from data_designer.engine.dataset_builders.scheduling.task_model import Task
from data_designer.engine.models.request_admission.resources import RequestDomain, RequestResourceKey

if TYPE_CHECKING:
    from data_designer.engine.column_generators.generators.base import ColumnGenerator


@dataclass(frozen=True)
class ResolvedTaskScheduling:
    """Scheduler inputs resolved from generator-facing metadata."""

    group: TaskGroupSpec
    resource_request: SchedulerResourceRequest
    request_resource_key: RequestResourceKey | None = None


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
        self._request_resource_limits = self._build_request_resource_limits()

    @property
    def diagnostics(self) -> tuple[dict[str, object], ...]:
        return tuple(self._diagnostics)

    @property
    def request_resource_limits(self) -> Mapping[SchedulerResourceKey, int]:
        return dict(self._request_resource_limits)

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
            request_resource_key=resolved.request_resource_key,
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
        request_resource_key = _request_resource_key(metadata)
        resource_request = {"submission": 1, "llm_wait": 1}
        if request_resource_key is not None:
            resource_request[request_scheduler_resource_key(request_resource_key)] = 1
        return ResolvedTaskScheduling(
            group=TaskGroupSpec(
                key=TaskGroupKey(kind=metadata.kind, identity=identity),
                weight=float(weight),
                admitted_limit=admitted_limit,
            ),
            resource_request=SchedulerResourceRequest(resource_request),
            request_resource_key=request_resource_key,
        )

    def _build_request_resource_limits(self) -> dict[SchedulerResourceKey, int]:
        limits: dict[SchedulerResourceKey, int] = {}
        for metadata in self._metadata_by_generator_id.values():
            resource = _request_resource_key(metadata)
            if resource is None:
                continue
            key = request_scheduler_resource_key(resource)
            cap = max(1, metadata.weight)
            limits[key] = min(limits.get(key, cap), cap)
        return limits


def _request_resource_key(metadata: SchedulingMetadata) -> RequestResourceKey | None:
    if metadata.kind != "model":
        return None
    _kind, provider_name, model_id, generation_kind = metadata.identity
    try:
        domain = RequestDomain(generation_kind)
    except ValueError:
        return None
    return RequestResourceKey(provider_name=provider_name, model_id=model_id, domain=domain)
