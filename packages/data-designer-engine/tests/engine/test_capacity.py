# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from data_designer.engine.capacity import (
    AsyncCapacityConfigured,
    AsyncCapacityObservedMaxima,
    AsyncCapacityPlan,
    AsyncCapacityRuntimeSnapshot,
    CapacityValue,
    RequestAdmissionConfigSnapshot,
    RowGroupAdmission,
)
from data_designer.engine.models.clients.request_admission import (
    ProviderModelKey,
    ProviderModelStaticCap,
    RequestAdmissionConfig,
    RequestDomain,
    RequestResourceKey,
)


def test_request_admission_config_snapshot_records_resources() -> None:
    resource = RequestResourceKey("nvidia", "nemotron", RequestDomain.CHAT)
    config = RequestAdmissionConfig(initial_limits={resource: 2}, max_limit_clamps={resource: 4})

    snapshot = RequestAdmissionConfigSnapshot.from_config(config)

    assert snapshot.resources == (resource,)
    assert snapshot.initial_limits[resource] == 2
    assert snapshot.max_limit_clamps[resource] == 4


def test_async_capacity_plan_records_configured_runtime_and_maxima() -> None:
    resource = RequestResourceKey("nvidia", "nemotron", RequestDomain.CHAT)
    provider_model = ProviderModelKey("nvidia", "nemotron")
    static_cap = ProviderModelStaticCap(cap=4, aliases=("default",), raw_caps={"default": 4})

    plan = AsyncCapacityPlan(
        configured=AsyncCapacityConfigured(
            buffer_size=CapacityValue(value=16, source="run_config"),
            row_group_admission=RowGroupAdmission(
                row_group_concurrency=CapacityValue(value=2, source="dataset_builder"),
                observed_in_flight=1,
            ),
            submission_capacity=CapacityValue(value=8, source="engine_internal_config"),
            task_resource_limits=CapacityValue(value={"submission": 8, "llm_wait": 4}, source="engine_internal_config"),
            request_resources=CapacityValue(value=(resource,), source="runtime_snapshot"),
            provider_model_static_caps=CapacityValue(value={provider_model: static_cap}, source="model_metadata"),
            request_domain_initial_limits=CapacityValue(value={resource: 2}, source="engine_internal_config"),
            request_admission_config=CapacityValue(
                value=RequestAdmissionConfigSnapshot.from_config(RequestAdmissionConfig(initial_limits={resource: 2})),
                source="engine_internal_config",
            ),
            transport_pool_limits=CapacityValue(value={provider_model: 8}, source="adapter_config"),
        ),
        runtime_snapshot=AsyncCapacityRuntimeSnapshot(
            request_domain_current_limits={resource: 2},
            request_domain_effective_max={resource: 4},
            request_domain_blocked_until={resource: None},
            provider_model_aggregate_in_flight={provider_model: 0},
        ),
        observed_maxima=AsyncCapacityObservedMaxima(
            row_groups_in_flight=1,
            request_in_flight_by_resource={resource: 2},
            provider_model_aggregate_in_flight={provider_model: 2},
        ),
    )

    assert plan.configured.provider_model_static_caps.value[provider_model].merge_rule == "min_same_endpoint"
    assert plan.runtime_snapshot.request_domain_current_limits[resource] == 2
    assert plan.observed_maxima.provider_model_aggregate_in_flight[provider_model] == 2
