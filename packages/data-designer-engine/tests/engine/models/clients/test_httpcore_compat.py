# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from contextlib import suppress
from types import SimpleNamespace

import httpcore
import httpx
import pytest

from data_designer.engine.models.clients.adapters.httpcore_compat import (
    _assign_http1_requests,
    install_linear_http1_assignment,
)


class _Connection:
    def __init__(
        self,
        origin: str,
        *,
        closed: bool = False,
        expired: bool = False,
        available: bool = True,
        idle: bool = True,
    ) -> None:
        self.origin = origin
        self.closed = closed
        self.expired = expired
        self.available = available
        self.idle = idle
        self.can_handle_calls = 0

    def is_closed(self) -> bool:
        return self.closed

    def has_expired(self) -> bool:
        return self.expired

    def is_available(self) -> bool:
        return self.available

    def is_idle(self) -> bool:
        return self.idle

    def can_handle_request(self, origin: str) -> bool:
        self.can_handle_calls += 1
        return self.origin == origin


class _PoolRequest:
    def __init__(self, origin: str) -> None:
        self.request = SimpleNamespace(url=SimpleNamespace(origin=origin))
        self.connection: _Connection | None = None

    def is_queued(self) -> bool:
        return self.connection is None

    def assign_to_connection(self, connection: _Connection) -> None:
        self.connection = connection


class _Pool:
    def __init__(
        self,
        connections: list[_Connection],
        requests: list[_PoolRequest],
        *,
        max_connections: int,
        max_keepalive_connections: int,
    ) -> None:
        self._connections = connections
        self._requests = requests
        self._max_connections = max_connections
        self._max_keepalive_connections = max_keepalive_connections
        self.created_connections: list[_Connection] = []

    def create_connection(self, origin: str) -> _Connection:
        connection = _Connection(origin)
        self.created_connections.append(connection)
        return connection


class _HTTP1Server:
    def __init__(self, *, close_connections: bool = False, response_delay_s: float = 0.0) -> None:
        self.close_connections = close_connections
        self.response_delay_s = response_delay_s
        self.connection_count = 0
        self.active_requests = 0
        self.max_active_requests = 0
        self._server: asyncio.AbstractServer | None = None
        self.url = ""

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._handle_connection, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"

    async def close(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _handle_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        self.connection_count += 1
        try:
            while True:
                try:
                    await reader.readuntil(b"\r\n\r\n")
                except (asyncio.IncompleteReadError, asyncio.LimitOverrunError, ConnectionError):
                    break

                self.active_requests += 1
                self.max_active_requests = max(self.max_active_requests, self.active_requests)
                try:
                    await asyncio.sleep(self.response_delay_s)
                    connection_header = b"close" if self.close_connections else b"keep-alive"
                    writer.write(
                        b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: " + connection_header + b"\r\n\r\nOK"
                    )
                    await writer.drain()
                finally:
                    self.active_requests -= 1

                if self.close_connections:
                    break
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()


def test_http1_assignment_does_not_assign_a_connection_twice() -> None:
    connections = [_Connection("origin"), _Connection("origin")]
    requests = [_PoolRequest("origin") for _ in range(3)]
    pool = _Pool(connections, requests, max_connections=2, max_keepalive_connections=2)

    closing = _assign_http1_requests(pool)

    assert closing == []
    assert {requests[0].connection, requests[1].connection} == set(connections)
    assert requests[2].connection is None


@pytest.mark.parametrize("connection", [_Connection("origin", expired=True), _Connection("other")])
def test_http1_assignment_replaces_unusable_connection(connection: _Connection) -> None:
    request = _PoolRequest("origin")
    pool = _Pool([connection], [request], max_connections=1, max_keepalive_connections=1)

    closing = _assign_http1_requests(pool)

    assert closing == [connection]
    assert request.connection is pool.created_connections[0]
    assert pool._connections == pool.created_connections


def test_http1_assignment_enforces_keepalive_limit() -> None:
    connections = [_Connection("origin") for _ in range(3)]
    pool = _Pool(connections, [], max_connections=3, max_keepalive_connections=1)

    closing = _assign_http1_requests(pool)

    assert pool._connections == connections[:1]
    assert closing == connections[1:]


def test_http1_assignment_scales_linearly_for_one_origin() -> None:
    connections = [_Connection("origin") for _ in range(128)]
    requests = [_PoolRequest("origin") for _ in range(128)]
    pool = _Pool(connections, requests, max_connections=128, max_keepalive_connections=128)

    _assign_http1_requests(pool)

    assert all(request.connection is not None for request in requests)
    assert sum(connection.can_handle_calls for connection in connections) == len(requests)


@pytest.mark.asyncio
async def test_install_is_scoped_and_idempotent() -> None:
    first = httpx.AsyncHTTPTransport()
    second = httpx.AsyncHTTPTransport()
    original_pool_method = httpcore.AsyncConnectionPool._assign_requests_to_connections
    try:
        assert install_linear_http1_assignment(first)
        assert install_linear_http1_assignment(first)
        assert first._pool._assign_requests_to_connections.__func__ is not original_pool_method
        assert second._pool._assign_requests_to_connections.__func__ is original_pool_method
        assert httpcore.AsyncConnectionPool._assign_requests_to_connections is original_pool_method
    finally:
        await first.aclose()
        await second.aclose()


@pytest.mark.asyncio
async def test_install_skips_unvalidated_httpcore_version(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = httpx.AsyncHTTPTransport()
    original_pool_method = transport._pool._assign_requests_to_connections.__func__
    monkeypatch.setattr(httpcore, "__version__", "1.0.10")
    try:
        assert not install_linear_http1_assignment(transport)
        assert transport._pool._assign_requests_to_connections.__func__ is original_pool_method
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_http2_pool_uses_original_assignment() -> None:
    transport = httpx.AsyncHTTPTransport(http2=True)
    original_pool_method = transport._pool._assign_requests_to_connections.__func__
    try:
        assert not install_linear_http1_assignment(transport)
        assert transport._pool._assign_requests_to_connections.__func__ is original_pool_method
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_proxy_pool_uses_original_assignment() -> None:
    transport = httpx.AsyncHTTPTransport(proxy="http://127.0.0.1:9")
    original_pool_method = transport._pool._assign_requests_to_connections.__func__
    try:
        assert not install_linear_http1_assignment(transport)
        assert transport._pool._assign_requests_to_connections.__func__ is original_pool_method
    finally:
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("close_connections", "expected_connection_count"), [(False, 1), (True, 6)])
async def test_http1_transport_handles_reuse_and_connection_close(
    close_connections: bool, expected_connection_count: int
) -> None:
    server = _HTTP1Server(close_connections=close_connections)
    await server.start()
    transport = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=2, max_keepalive_connections=1))
    assert install_linear_http1_assignment(transport)
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            responses = [await client.get(server.url) for _ in range(6)]
        assert [response.status_code for response in responses] == [200] * 6
        assert server.connection_count == expected_connection_count
    finally:
        await server.close()


@pytest.mark.asyncio
async def test_http1_transport_preserves_connection_capacity() -> None:
    server = _HTTP1Server(response_delay_s=0.01)
    await server.start()
    transport = httpx.AsyncHTTPTransport(limits=httpx.Limits(max_connections=2, max_keepalive_connections=2))
    assert install_linear_http1_assignment(transport)
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            responses = await asyncio.gather(*(client.get(server.url) for _ in range(8)))
        assert [response.status_code for response in responses] == [200] * 8
        assert server.connection_count == 2
        assert server.max_active_requests == 2
    finally:
        await server.close()
