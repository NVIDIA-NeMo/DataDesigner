# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from conftest import RuntimeCase

from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.ports import allocation_ports, resolve_allocation_deployments


def test_allocation_ports_are_deterministic_and_isolated_by_gpu(runtime_case: RuntimeCase) -> None:
    first = allocation_ports(runtime_case.context, {"SLURM_JOB_GPUS": "0"})
    second = allocation_ports(runtime_case.context, {"SLURM_JOB_GPUS": "1"})

    assert first == allocation_ports(runtime_case.context, {"SLURM_JOB_GPUS": "0"})
    assert set(first).isdisjoint(second)
    assert all(10000 <= port < 10256 for port in first)
    assert all(10256 <= port < 10512 for port in second)


def test_allocation_deployment_uses_remapped_ports(runtime_case: RuntimeCase) -> None:
    environment = {"SLURM_JOB_GPUS": "0"}
    deployment = resolve_allocation_deployments(runtime_case.context, environment)[0]
    ports = set(allocation_ports(runtime_case.context, environment))

    assert deployment.logical_endpoint.port in ports
    assert {backend.port for backend in deployment.backend_endpoints} <= ports
    assert {probe.port for probe in deployment.readiness_probes} <= ports
    assert {process.http_port for process in deployment.processes if process.http_port is not None} <= ports


def test_allocation_ports_skip_client_otel_port(runtime_case: RuntimeCase) -> None:
    environment = {"SLURM_JOB_GPUS": "0"}
    otel_port = allocation_ports(runtime_case.context, environment)[0]
    invocation = runtime_case.context.plan.invocation.model_copy(
        update={"effective_run_config": {"otel_metrics_port": otel_port}}
    )
    plan = runtime_case.context.plan.model_copy(update={"invocation": invocation})
    context = AllocationContext(
        plan=plan,
        shard=runtime_case.context.shard,
        attempt=runtime_case.context.attempt,
        attempt_directory=runtime_case.context.attempt_directory,
    )

    assert otel_port not in allocation_ports(context, environment)


@pytest.mark.parametrize("environment", ({}, {"SLURM_JOB_GPUS": ""}, {"SLURM_JOB_GPUS": "0,0"}))
def test_allocation_ports_reject_invalid_gpu_ids(
    runtime_case: RuntimeCase,
    environment: dict[str, str],
) -> None:
    with pytest.raises(SlurmRuntimeError, match="SLURM_JOB_GPUS"):
        allocation_ports(runtime_case.context, environment)


def test_visible_gpu_mode_keeps_planned_ports(runtime_case: RuntimeCase) -> None:
    profile = runtime_case.context.plan.selected_profile.profile.model_copy(update={"gpu_request_mode": "visible"})
    selected_profile = runtime_case.context.plan.selected_profile.model_copy(update={"profile": profile})
    plan = runtime_case.context.plan.model_copy(update={"selected_profile": selected_profile})
    context = AllocationContext(
        plan=plan,
        shard=runtime_case.context.shard,
        attempt=runtime_case.context.attempt,
        attempt_directory=runtime_case.context.attempt_directory,
    )
    planned = tuple(port.port for port in plan.client.ports) + tuple(
        port.port for deployment in plan.deployments for port in deployment.ports
    )

    assert allocation_ports(context, {}) == planned
