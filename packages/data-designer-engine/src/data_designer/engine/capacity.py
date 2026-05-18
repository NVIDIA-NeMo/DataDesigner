# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

from data_designer.engine.dataset_builders.scheduling.resources import SchedulerResourceKey, TaskGroupKey
from data_designer.engine.models.request_admission.config import RequestAdmissionConfig
from data_designer.engine.models.request_admission.resources import RequestResourceKey
from data_designer.engine.models.resources import ProviderModelKey, ProviderModelStaticCap

_T = TypeVar("_T")

CapacityValueSource = Literal[
    "default",
    "run_config",
    "dataset_builder",
    "model_metadata",
    "engine_internal_config",
    "adapter_config",
    "environment",
    "runtime_snapshot",
    "benchmark_override",
]


@dataclass(frozen=True)
class CapacityValue(Generic[_T]):
    value: _T | None
    source: CapacityValueSource
    fallback_from: str | None = None
    missing_reason: str | None = None


@dataclass(frozen=True)
class RowGroupAdmission:
    row_group_concurrency: CapacityValue[int]
    observed_in_flight: int | None = None


@dataclass(frozen=True)
class RequestAdmissionConfigSnapshot:
    resources: Sequence[RequestResourceKey]
    initial_limits: Mapping[RequestResourceKey, int]
    max_limit_clamps: Mapping[RequestResourceKey, int | None]
    cooldown_seconds: float
    multiplicative_decrease_factor: float
    additive_increase_step: int
    increase_after_successes: int
    default_queue_wait_timeout_seconds: float | None

    @classmethod
    def from_config(cls, config: RequestAdmissionConfig) -> RequestAdmissionConfigSnapshot:
        resources = tuple(sorted({*config.initial_limits, *config.max_limit_clamps}))
        return cls(
            resources=resources,
            initial_limits=dict(config.initial_limits),
            max_limit_clamps=dict(config.max_limit_clamps),
            cooldown_seconds=config.cooldown_seconds,
            multiplicative_decrease_factor=config.multiplicative_decrease_factor,
            additive_increase_step=config.additive_increase_step,
            increase_after_successes=config.increase_after_successes,
            default_queue_wait_timeout_seconds=config.default_queue_wait_timeout_seconds,
        )


@dataclass(frozen=True)
class AsyncCapacityConfigured:
    buffer_size: CapacityValue[int]
    row_group_admission: RowGroupAdmission
    submission_capacity: CapacityValue[int]
    task_resource_limits: CapacityValue[Mapping[SchedulerResourceKey, int]]
    request_resources: CapacityValue[Sequence[RequestResourceKey]]
    provider_model_static_caps: CapacityValue[Mapping[ProviderModelKey, ProviderModelStaticCap]]
    request_domain_initial_limits: CapacityValue[Mapping[RequestResourceKey, int]]
    request_admission_config: CapacityValue[RequestAdmissionConfigSnapshot]
    transport_pool_limits: CapacityValue[Mapping[ProviderModelKey, int]]


@dataclass(frozen=True)
class AsyncCapacityRuntimeSnapshot:
    request_domain_current_limits: Mapping[RequestResourceKey, int] | None = None
    request_domain_effective_max: Mapping[RequestResourceKey, int] | None = None
    request_domain_blocked_until: Mapping[RequestResourceKey, float | None] | None = None
    provider_model_aggregate_in_flight: Mapping[ProviderModelKey, int] | None = None


@dataclass(frozen=True)
class AsyncCapacityObservedMaxima:
    row_groups_in_flight: int = 0
    queued_tasks_by_group: Mapping[TaskGroupKey | str, int] = field(default_factory=dict)
    task_leases_by_resource: Mapping[SchedulerResourceKey, int] = field(default_factory=dict)
    request_waiters_by_resource: Mapping[RequestResourceKey, int] = field(default_factory=dict)
    request_in_flight_by_resource: Mapping[RequestResourceKey, int] = field(default_factory=dict)
    provider_model_aggregate_in_flight: Mapping[ProviderModelKey, int] = field(default_factory=dict)
    request_domain_current_limits: Mapping[RequestResourceKey, int] = field(default_factory=dict)
    transport_pool_utilization: Mapping[ProviderModelKey, int] | None = None


@dataclass(frozen=True)
class AsyncCapacityPlan:
    configured: AsyncCapacityConfigured
    runtime_snapshot: AsyncCapacityRuntimeSnapshot
    observed_maxima: AsyncCapacityObservedMaxima


def missing_capacity_value(
    *,
    source: CapacityValueSource,
    missing_reason: str,
    fallback_from: str | None = None,
) -> CapacityValue[object]:
    return CapacityValue(value=None, source=source, fallback_from=fallback_from, missing_reason=missing_reason)
