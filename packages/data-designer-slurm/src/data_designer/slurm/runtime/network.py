# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Shared validation for scheduler-derived runtime network identities."""

from __future__ import annotations

import re

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


__all__ = ["validate_host_name", "validate_network_port"]
