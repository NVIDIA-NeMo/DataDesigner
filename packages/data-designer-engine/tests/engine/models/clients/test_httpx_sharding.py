# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from collections.abc import Callable
from contextlib import suppress
from unittest.mock import AsyncMock

import httpx
import pytest

from data_designer.engine.models.clients.adapters.httpx_sharding import ShardedAsyncHTTPTransport
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport


class _RecordingTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        index: int,
        *,
        delay_s: float = 0.0,
        error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.index = index
        self.delay_s = delay_s
        self.error = error
        self.close_error = close_error
        self.requests: list[httpx.Request] = []
        self.active = 0
        self.max_active = 0
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        try:
            await asyncio.sleep(self.delay_s)
            if self.error is not None:
                raise self.error
            return httpx.Response(200, json={"shard": self.index}, request=request)
        finally:
            self.active -= 1

    async def aclose(self) -> None:
        self.closed = True
        if self.close_error is not None:
            raise self.close_error


class _HTTP1Server:
    def __init__(self, *, close_connections: bool) -> None:
        self.close_connections = close_connections
        self.connection_count = 0
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
                connection_header = b"close" if self.close_connections else b"keep-alive"
                writer.write(
                    b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: " + connection_header + b"\r\n\r\nOK"
                )
                await writer.drain()
                if self.close_connections:
                    break
        finally:
            writer.close()
            with suppress(ConnectionError):
                await writer.wait_closed()


def _recording_factory(
    transports: list[_RecordingTransport],
    *,
    delays: tuple[float, ...] = (),
    errors: dict[int, Exception] | None = None,
    close_errors: dict[int, Exception] | None = None,
) -> Callable[[httpx.Limits], httpx.AsyncBaseTransport]:
    def create(_: httpx.Limits) -> httpx.AsyncBaseTransport:
        index = len(transports)
        transport = _RecordingTransport(
            index,
            delay_s=delays[index] if delays else 0.0,
            error=(errors or {}).get(index),
            close_error=(close_errors or {}).get(index),
        )
        transports.append(transport)
        return transport

    return create


@pytest.mark.parametrize(
    ("maximum", "keepalive"),
    [(16_384, 8_192), (600, 300), (32, 16)],
)
def test_shards_preserve_aggregate_limits(maximum: int, keepalive: int) -> None:
    captured_limits: list[httpx.Limits] = []

    def create(limits: httpx.Limits) -> httpx.AsyncBaseTransport:
        captured_limits.append(limits)
        return httpx.MockTransport(lambda request: httpx.Response(200, request=request))

    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=maximum, max_keepalive_connections=keepalive, keepalive_expiry=7.0),
        transport_factory=create,
    )

    assert len(captured_limits) == 16
    assert sum(limit.max_connections or 0 for limit in captured_limits) == maximum
    assert sum(limit.max_keepalive_connections or 0 for limit in captured_limits) == keepalive
    assert all(maximum // 16 <= limit.max_connections <= (maximum + 15) // 16 for limit in captured_limits)
    assert all(
        keepalive // 16 <= limit.max_keepalive_connections <= (keepalive + 15) // 16 for limit in captured_limits
    )
    assert all(limit.keepalive_expiry == 7.0 for limit in captured_limits)
    assert transport.shard_limits == tuple(captured_limits)


@pytest.mark.parametrize(
    ("shard_count", "maximum", "keepalive", "message"),
    [
        (0, 32, 16, "shard_count must be positive"),
        (-1, 32, 16, "shard_count must be positive"),
        (16, None, 16, "finite connection limits"),
        (16, 32, None, "finite connection limits"),
        (16, 8, 16, "at least one connection per shard"),
        (16, 32, 8, "at least one connection per shard"),
    ],
)
def test_invalid_limits_rejected(shard_count: int, maximum: int | None, keepalive: int | None, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ShardedAsyncHTTPTransport(
            limits=httpx.Limits(max_connections=maximum, max_keepalive_connections=keepalive),
            shard_count=shard_count,
        )


@pytest.mark.asyncio
async def test_closed_transport_rejects_requests() -> None:
    transports: list[_RecordingTransport] = []
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
        transport_factory=_recording_factory(transports),
    )
    await transport.aclose()

    with pytest.raises(RuntimeError, match="Transport is closed"):
        await transport.handle_async_request(httpx.Request("GET", "https://example.test"))
    assert all(shard.closed and not shard.requests for shard in transports)


@pytest.mark.asyncio
async def test_round_robin_distribution_with_variable_request_durations() -> None:
    transports: list[_RecordingTransport] = []
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=16, max_keepalive_connections=8),
        shard_count=4,
        transport_factory=_recording_factory(transports, delays=(0.04, 0.0, 0.02, 0.01)),
    )
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            responses = await asyncio.gather(*(client.get(f"https://example.test/{index}") for index in range(16)))

        assert [response.json()["shard"] for response in responses] == [0, 1, 2, 3] * 4
        assert [len(shard.requests) for shard in transports] == [4, 4, 4, 4]
        assert transports[0].max_active == 4
        assert transports[1].max_active >= 1
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_response_exception_and_timeout_propagation() -> None:
    transports: list[_RecordingTransport] = []
    error = httpx.ConnectError("unavailable")
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        shard_count=2,
        transport_factory=_recording_factory(transports, errors={1: error}),
    )
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            response = await client.get("https://example.test/ok", timeout=3.0)
            with pytest.raises(httpx.ConnectError, match="unavailable"):
                await client.get("https://example.test/error")

        assert response.json() == {"shard": 0}
        assert transports[0].requests[0].extensions["timeout"] == {
            "connect": 3.0,
            "read": 3.0,
            "write": 3.0,
            "pool": 3.0,
        }
    finally:
        await transport.aclose()


@pytest.mark.asyncio
async def test_retry_transport_retries_on_the_next_shard() -> None:
    attempts = 0
    shard_requests = [0, 0]
    created_shards = 0

    def create(_: httpx.Limits) -> httpx.AsyncBaseTransport:
        nonlocal created_shards
        index = created_shards
        created_shards += 1

        async def handle(request: httpx.Request) -> httpx.Response:
            nonlocal attempts
            shard_requests[index] += 1
            attempts += 1
            return httpx.Response(503 if attempts == 1 else 200, request=request)

        return httpx.MockTransport(handle)

    sharded = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        shard_count=2,
        transport_factory=create,
    )
    retrying = create_retry_transport(
        RetryConfig(max_retries=1, backoff_factor=0, backoff_jitter=0),
        transport=sharded,
    )
    async with httpx.AsyncClient(transport=retrying) as client:
        response = await client.get("https://example.test/retry")

    assert response.status_code == 200
    assert attempts == 2
    assert shard_requests == [1, 1]


@pytest.mark.asyncio
async def test_close_attempts_every_child_and_propagates_error() -> None:
    transports: list[_RecordingTransport] = []
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=6, max_keepalive_connections=3),
        shard_count=3,
        transport_factory=_recording_factory(transports, close_errors={1: RuntimeError("close failed")}),
    )

    with pytest.raises(RuntimeError, match="close failed"):
        await transport.aclose()

    assert all(shard.closed for shard in transports)
    with pytest.raises(RuntimeError, match="close failed"):
        await transport.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize(("close_connections", "expected_connections"), [(False, 2), (True, 8)])
async def test_shards_handle_keepalive_and_connection_close(close_connections: bool, expected_connections: int) -> None:
    server = _HTTP1Server(close_connections=close_connections)
    await server.start()
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        shard_count=2,
    )
    try:
        async with httpx.AsyncClient(transport=transport) as client:
            responses = [await client.get(server.url) for _ in range(8)]
        assert [response.status_code for response in responses] == [200] * 8
        assert server.connection_count == expected_connections
    finally:
        await server.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_before_retry", [False, True])
async def test_close_survives_cancellation_and_waits_for_every_child(fail_before_retry: bool) -> None:
    transports: list[_RecordingTransport] = []
    transport = ShardedAsyncHTTPTransport(
        limits=httpx.Limits(max_connections=4, max_keepalive_connections=2),
        shard_count=2,
        transport_factory=_recording_factory(transports),
    )
    started = asyncio.Event()
    release = asyncio.Event()
    finished = asyncio.Event()

    async def close() -> None:
        started.set()
        await release.wait()
        transports[0].closed = True
        finished.set()
        if fail_before_retry:
            raise RuntimeError("close failed")

    transports[0].aclose = AsyncMock(side_effect=close)
    closing = asyncio.create_task(transport.aclose())
    await asyncio.wait_for(started.wait(), timeout=5)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing
    if fail_before_retry:
        release.set()
        await asyncio.wait_for(finished.wait(), timeout=5)
        await asyncio.sleep(0)  # Let the shared gather record the completed close.
        for _ in range(2):
            with pytest.raises(RuntimeError, match="close failed"):
                await transport.aclose()
        assert all(shard.closed for shard in transports)
        transports[0].aclose.assert_awaited_once()
        return
    retry = asyncio.create_task(transport.aclose())
    try:
        await asyncio.sleep(0)
        assert not retry.done()
    finally:
        release.set()
        await asyncio.wait_for(retry, timeout=5)
    assert all(shard.closed for shard in transports)
    transports[0].aclose.assert_awaited_once()
