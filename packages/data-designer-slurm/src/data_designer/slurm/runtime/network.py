# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for scheduler-derived runtime network identities."""

from __future__ import annotations

import re
from pathlib import Path

from data_designer.slurm.runtime.errors import SlurmRuntimeError, SlurmRuntimeErrorCode

_PORT_RANGE_START = 10000
_PORT_RANGE_END = 65536
_SYSCTL_ROOT = Path("/proc/sys") / "net" / "ipv4"
_HOST_NAME_PATTERN = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9._-]{0,251}[A-Za-z0-9])?$")


def validate_host_name(host: str) -> str:
    """Return a safe scheduler host name or raise ``ValueError``."""
    if type(host) is not str or _HOST_NAME_PATTERN.fullmatch(host) is None:
        raise ValueError("allocation host identity is invalid")
    return host


def validate_network_port(port: int) -> int:
    """Return a valid TCP port or raise ``ValueError``."""
    if type(port) is not int or not 1 <= port <= 65535:
        raise ValueError("allocation network port is invalid")
    return port


def get_service_ports() -> tuple[int, ...]:
    """Return unprivileged ports excluded from automatic TCP source allocation.

    Preserve the preferred service range where safe, then use lower ports.
    Kernel-reserved ports remain eligible but must still pass bind preflight.
    """
    try:
        ephemeral_start, ephemeral_end = map(int, (_SYSCTL_ROOT / "ip_local_port_range").read_text().split())
        try:
            unprivileged_start = int((_SYSCTL_ROOT / "ip_unprivileged_port_start").read_text())
        except FileNotFoundError:
            # Kernels without this sysctl use a fixed privileged-port boundary.
            unprivileged_start = 1024
        if not 1 <= ephemeral_start <= ephemeral_end < _PORT_RANGE_END or not 0 <= unprivileged_start < _PORT_RANGE_END:
            raise ValueError("invalid port range")
        reserved: set[int] = set()
        for field in (_SYSCTL_ROOT / "ip_local_reserved_ports").read_text().strip().split(","):
            if not field:
                continue
            bounds = tuple(map(int, field.split("-")))
            if len(bounds) not in (1, 2) or not 1 <= bounds[0] <= bounds[-1] < _PORT_RANGE_END:
                raise ValueError("invalid reserved port range")
            reserved.update(range(bounds[0], bounds[-1] + 1))
    except (OSError, ValueError) as error:
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "cannot read effective TCP ephemeral, reserved, and unprivileged port settings",
        ) from error
    minimum = max(1024, unprivileged_start)
    ports = tuple(
        port
        for port in range(minimum, _PORT_RANGE_END)
        if not ephemeral_start <= port <= ephemeral_end or port in reserved
    )
    return tuple(port for port in ports if port >= _PORT_RANGE_START) + tuple(
        port for port in ports if port < _PORT_RANGE_START
    )


def verify_service_ports(ports: tuple[int, ...]) -> None:
    """Reject service ports unsafe under this node's effective TCP settings."""
    available = set(get_service_ports())
    if any(port not in available for port in ports):
        raise SlurmRuntimeError(
            SlurmRuntimeErrorCode.PREFLIGHT_FAILED,
            "allocation ports overlap effective ephemeral or privileged ports on this node",
        )


__all__ = ["get_service_ports", "validate_host_name", "validate_network_port", "verify_service_ports"]
