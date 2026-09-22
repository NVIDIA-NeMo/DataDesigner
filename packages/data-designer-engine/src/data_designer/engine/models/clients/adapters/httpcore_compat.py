# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Scoped compatibility for httpcore 1.0.9 async HTTP/1 pools."""

from __future__ import annotations

import importlib
import logging
from types import MethodType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx


logger = logging.getLogger(__name__)

# Remove when the upstream fix is released and adopted as the minimum version:
# https://github.com/encode/httpcore/pull/1035
_LINEAR_ASSIGNMENT_HTTPCORE_VERSION = "1.0.9"
_LINEAR_ASSIGNMENT_MARKER = "_data_designer_linear_http1_assignment"
_REQUIRED_POOL_ATTRIBUTES = (
    "_assign_requests_to_connections",
    "_connections",
    "_http1",
    "_http2",
    "_max_connections",
    "_max_keepalive_connections",
    "_requests",
    "create_connection",
)


def _use_default_assignment(reason: str) -> bool:
    logger.warning("Using default httpcore request assignment: %s", reason)
    return False


def install_linear_http1_assignment(transport: httpx.AsyncHTTPTransport) -> bool:
    """Install linear assignment on one single-origin HTTP/1 transport pool."""
    try:
        httpcore = importlib.import_module("httpcore")
    except ImportError:
        return _use_default_assignment("httpcore is unavailable")

    version = getattr(httpcore, "__version__", None)
    if version != _LINEAR_ASSIGNMENT_HTTPCORE_VERSION:
        return _use_default_assignment(
            f"httpcore {version or 'unknown'} is not the validated {_LINEAR_ASSIGNMENT_HTTPCORE_VERSION} release"
        )

    pool = getattr(transport, "_pool", None)
    pool_type = getattr(httpcore, "AsyncConnectionPool", None)
    # Proxy pool subclasses have not been validated against this algorithm.
    if pool_type is None or type(pool) is not pool_type:
        return _use_default_assignment(f"pool type {type(pool).__name__} is not validated")

    missing_attributes = [attribute for attribute in _REQUIRED_POOL_ATTRIBUTES if not hasattr(pool, attribute)]
    if missing_attributes:
        return _use_default_assignment(f"pool is missing private attributes: {', '.join(missing_attributes)}")

    if (
        not isinstance(pool._connections, list)
        or not isinstance(pool._requests, list)
        or not isinstance(pool._max_connections, int)
        or not isinstance(pool._max_keepalive_connections, int)
        or not isinstance(pool._http1, bool)
        or not isinstance(pool._http2, bool)
        or not callable(pool._assign_requests_to_connections)
        or not callable(pool.create_connection)
    ):
        return _use_default_assignment("pool private attributes have an incompatible shape")

    if not pool._http1 or pool._http2:
        return _use_default_assignment("pool is not HTTP/1-only")

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

    idle_connections_to_close = max(
        0,
        sum(connection.is_idle() for connection in available_connections) - pool._max_keepalive_connections,
    )
    if idle_connections_to_close:
        kept_connections = []
        for connection in available_connections:
            if idle_connections_to_close and connection.is_idle():
                closing_connections.append(connection)
                idle_connections_to_close -= 1
            else:
                kept_connections.append(connection)
        available_connections = kept_connections

    pool._connections = available_connections + occupied_connections
    return closing_connections
