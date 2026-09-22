# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped compatibility for httpcore 1.0.9 async HTTP/1 pools."""

from __future__ import annotations

import importlib
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


_LINEAR_ASSIGNMENT_HTTPCORE_VERSION = "1.0.9"
_LINEAR_ASSIGNMENT_MARKER = "_data_designer_linear_http1_assignment"


def install_linear_http1_assignment(transport: httpx.AsyncHTTPTransport) -> bool:
    """Install linear assignment on one single-origin HTTP/1 transport pool."""
    try:
        httpcore = importlib.import_module("httpcore")
    except ImportError:
        return False

    if getattr(httpcore, "__version__", None) != _LINEAR_ASSIGNMENT_HTTPCORE_VERSION:
        return False

    pool = getattr(transport, "_pool", None)
    pool_type = getattr(httpcore, "AsyncConnectionPool", None)
    # Proxy pool subclasses have not been validated against this algorithm.
    if pool_type is None or type(pool) is not pool_type or getattr(pool, "_http2", False):
        return False

    original_assign = pool._assign_requests_to_connections
    assignment_function = getattr(original_assign, "__func__", original_assign)
    if getattr(assignment_function, _LINEAR_ASSIGNMENT_MARKER, False) is True:
        return True

    def assign_requests_to_connections(self: Any) -> list[Any]:
        return _assign_http1_requests(self)

    setattr(assign_requests_to_connections, _LINEAR_ASSIGNMENT_MARKER, True)
    pool._assign_requests_to_connections = MethodType(assign_requests_to_connections, pool)
    return True


def _assign_http1_requests(pool: Any) -> list[Any]:
    """Assign once per connection, count only idle keepalives, and evict oldest first."""
    closing_connections = []
    available_connections = []
    occupied_connections = []

    for connection in pool._connections:
        if connection.is_closed():
            continue
        if connection.has_expired():
            closing_connections.append(connection)
        elif connection.is_available():
            available_connections.append(connection)
        elif connection.is_idle():
            # Defensive for nonstandard HTTP/1 connections where idle is not available.
            closing_connections.append(connection)
        else:
            occupied_connections.append(connection)

    new_connections_remaining = pool._max_connections - len(available_connections) - len(occupied_connections)

    for pool_request in pool._requests:
        if not pool_request.is_queued():
            continue
        origin = pool_request.request.url.origin

        for index in range(len(available_connections) - 1, -1, -1):
            connection = available_connections[index]
            if connection.can_handle_request(origin):
                pool_request.assign_to_connection(connection)
                available_connections.pop(index)
                occupied_connections.append(connection)
                break
        else:
            if new_connections_remaining > 0:
                connection = pool.create_connection(origin)
                pool_request.assign_to_connection(connection)
                occupied_connections.append(connection)
                new_connections_remaining -= 1
                continue

            for index, connection in enumerate(available_connections):
                if connection.is_idle():
                    closing_connections.append(available_connections.pop(index))
                    connection = pool.create_connection(origin)
                    pool_request.assign_to_connection(connection)
                    occupied_connections.append(connection)
                    break
            else:
                break

    if len(available_connections) > pool._max_keepalive_connections:
        kept_connections = []
        idle_connections_kept = 0
        for connection in available_connections:
            if connection.is_idle():
                if idle_connections_kept >= pool._max_keepalive_connections:
                    closing_connections.append(connection)
                else:
                    kept_connections.append(connection)
                    idle_connections_kept += 1
            else:
                kept_connections.append(connection)
        available_connections = kept_connections

    pool._connections = available_connections + occupied_connections
    return closing_connections
