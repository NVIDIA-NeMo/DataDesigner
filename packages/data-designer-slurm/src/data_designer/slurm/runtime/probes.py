# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bounded loopback readiness probes for allocation-local endpoints."""

from __future__ import annotations

import http.client
from typing import Protocol


class ReadinessProber(Protocol):
    """Probe one loopback HTTP endpoint."""

    def is_ready(self, host: str, port: int, path: str, *, timeout_seconds: float) -> bool:
        """Return whether the endpoint produced HTTP 200 within the timeout."""
        ...


class HttpReadinessProber:
    """Production stdlib HTTP prober with no redirect or proxy behavior."""

    def is_ready(self, host: str, port: int, path: str, *, timeout_seconds: float) -> bool:
        """Probe a reviewed loopback address and fully close the connection."""
        if host != "127.0.0.1" or not path.startswith("/") or timeout_seconds <= 0:
            return False
        connection = http.client.HTTPConnection(host, port, timeout=timeout_seconds)
        try:
            connection.request("GET", path, headers={"Connection": "close"})
            response = connection.getresponse()
            return response.status == 200
        except (OSError, http.client.HTTPException):
            return False
        finally:
            connection.close()


__all__ = ["HttpReadinessProber", "ReadinessProber"]
