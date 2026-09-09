# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from conftest import RuntimeCase

from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.ports import allocation_ports, resolve_allocation_deployments


def test_allocation_ports_are_deterministic_and_isolated_by_scheduler_identity(runtime_case: RuntimeCase) -> None:
    first = allocation_ports(runtime_case.context)
    scheduler = runtime_case.context.attempt.scheduler
    assert scheduler is not None
    alternate_attempt = runtime_case.context.attempt.model_copy(
        update={"scheduler": scheduler.model_copy(update={"array_job_id": 4102})}
    )
    alternate = AllocationContext(
        plan=runtime_case.context.plan,
        shard=runtime_case.context.shard,
        attempt=alternate_attempt,
        attempt_directory=runtime_case.context.attempt_directory,
    )

    assert first == allocation_ports(runtime_case.context)
    assert set(first).isdisjoint(allocation_ports(alternate))
    assert all(10000 <= port < 30000 for port in first)


def test_allocation_deployment_uses_remapped_ports(runtime_case: RuntimeCase) -> None:
    deployment = resolve_allocation_deployments(runtime_case.context)[0]
    ports = set(allocation_ports(runtime_case.context))

    assert deployment.logical_endpoint.port in ports
    assert {backend.port for backend in deployment.backend_endpoints} <= ports
    assert {probe.port for probe in deployment.readiness_probes} <= ports
    assert {process.http_port for process in deployment.processes if process.http_port is not None} <= ports


def test_allocation_ports_skip_client_otel_port(runtime_case: RuntimeCase) -> None:
    otel_port = allocation_ports(runtime_case.context)[0]
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

    assert otel_port not in allocation_ports(context)
