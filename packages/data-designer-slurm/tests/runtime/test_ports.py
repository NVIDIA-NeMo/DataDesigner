# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from conftest import RuntimeCase

from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.runtime.errors import SlurmRuntimeError
from data_designer.slurm.runtime.models import AllocationContext
from data_designer.slurm.runtime.network import get_service_ports, verify_service_ports
from data_designer.slurm.runtime.ports import (
    allocation_ports,
    resolve_allocation_deployments,
    resolve_allocation_plan,
)


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


@pytest.mark.parametrize(
    ("ephemeral", "reserved", "unprivileged", "first"),
    (
        ("9000 65000", "", "1024", 65001),
        ("9000 65535", "", "1024", 1024),
        ("1024 65535", "10000-10255,11000", "1024", 10000),
        ("10000 10255", "", "11000", 11000),
        ("1024 65534", "", "0", 65535),
    ),
)
def test_service_ports_respect_effective_kernel_ranges(
    port_sysctls: Path, ephemeral: str, reserved: str, unprivileged: str, first: int
) -> None:
    (port_sysctls / "ip_local_port_range").write_text(ephemeral)
    (port_sysctls / "ip_local_reserved_ports").write_text(reserved)
    (port_sysctls / "ip_unprivileged_port_start").write_text(unprivileged)

    ports = get_service_ports()

    assert ports[0] == first
    assert len(ports) == len(set(ports))
    assert all(port >= max(1024, int(unprivileged)) for port in ports)
    if reserved:
        assert set(ports) == {*range(10000, 10256), 11000}
    else:
        start, end = map(int, ephemeral.split())
        assert all(not start <= port <= end for port in ports)


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("ip_local_port_range", ""),
        ("ip_local_port_range", "0 65535"),
        ("ip_local_port_range", "65000 9000"),
        ("ip_local_port_range", None),
        ("ip_local_reserved_ports", "10000-9999"),
        ("ip_local_reserved_ports", "10000-10001-10002"),
        ("ip_local_reserved_ports", "65536"),
        ("ip_local_reserved_ports", None),
        ("ip_unprivileged_port_start", "invalid"),
        ("ip_unprivileged_port_start", "65536"),
    ),
)
def test_service_ports_fail_closed_on_unavailable_kernel_settings(
    port_sysctls: Path, name: str, value: str | None
) -> None:
    path = port_sysctls / name
    if value is None:
        path.unlink()
    else:
        path.write_text(value)

    with pytest.raises(SlurmRuntimeError, match="cannot read effective TCP"):
        get_service_ports()


@pytest.mark.parametrize("ephemeral", ("32768 60999", "9000 65000"))
def test_service_ports_support_missing_unprivileged_threshold(port_sysctls: Path, ephemeral: str) -> None:
    (port_sysctls / "ip_local_port_range").write_text(ephemeral)
    expected = get_service_ports()
    (port_sysctls / "ip_unprivileged_port_start").unlink()

    assert get_service_ports() == expected


def test_service_ports_reject_unreadable_unprivileged_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    read_text = Path.read_text

    def read(path: Path, *args: object, **kwargs: object) -> str:
        if path.name == "ip_unprivileged_port_start":
            raise PermissionError("injected unreadable setting")
        return read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)

    with pytest.raises(SlurmRuntimeError, match="cannot read effective TCP"):
        get_service_ports()


def test_allocation_ports_remain_isolated_across_ephemeral_gap(runtime_case: RuntimeCase, port_sysctls: Path) -> None:
    (port_sysctls / "ip_local_port_range").write_text("9000 65000")
    digest = runtime_case.context.plan.compute_sha256()
    blocks = [allocation_ports(runtime_case.context, {"SLURM_JOB_GPUS": str(gpu)}) for gpu in range(4)]

    assert len(set().union(*blocks)) == sum(map(len, blocks))
    assert all(not 9000 <= port <= 65000 for block in blocks for port in block)
    assert runtime_case.context.plan.compute_sha256() == digest


@pytest.mark.parametrize("gpu", ("0", "1", "1000"))
def test_allocation_ports_fail_when_safe_gpu_block_is_exhausted(
    runtime_case: RuntimeCase, port_sysctls: Path, gpu: str
) -> None:
    (port_sysctls / "ip_local_port_range").write_text("1024 65534")

    with pytest.raises(SlurmRuntimeError, match="available non-ephemeral"):
        allocation_ports(runtime_case.context, {"SLURM_JOB_GPUS": gpu})


def test_visible_mode_rejects_unsafe_planned_ports(runtime_case: RuntimeCase, port_sysctls: Path) -> None:
    (port_sysctls / "ip_local_port_range").write_text("1024 65535")
    profile = runtime_case.context.plan.selected_profile.profile.model_copy(update={"gpu_request_mode": "visible"})
    selected = runtime_case.context.plan.selected_profile.model_copy(update={"profile": profile})
    plan = runtime_case.context.plan.model_copy(update={"selected_profile": selected})

    with pytest.raises(SlurmRuntimeError, match="overlap effective ephemeral"):
        resolve_allocation_plan(plan, {})


def test_remote_node_checks_its_own_port_policy(multi_node_plan: ResolvedSlurmRunPlan, port_sysctls: Path) -> None:
    plan = resolve_allocation_plan(multi_node_plan, {"SLURM_JOB_GPUS": "0"})
    remote_ports = tuple(
        port.port for deployment in plan.deployments for port in deployment.ports if port.node_index > 0
    )
    assert remote_ports
    verify_service_ports(remote_ports)
    (port_sysctls / "ip_local_port_range").write_text("10000 20000")

    with pytest.raises(SlurmRuntimeError, match="on this node"):
        verify_service_ports(remote_ports)
