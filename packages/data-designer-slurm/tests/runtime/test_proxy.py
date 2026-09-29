# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import asyncio
import socket
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

import pytest
from aiohttp import ClientSession, web
from aiohttp.test_utils import TestServer

from data_designer.slurm.runtime import proxy as runtime_proxy
from data_designer.slurm.runtime.proxy import _Backend, _BackendPool, _parse_backend, _ProxyApplication

_Handler = Callable[[web.Request], Awaitable[web.StreamResponse]]


@asynccontextmanager
async def _serve(application: web.Application) -> AsyncIterator[TestServer]:
    server = TestServer(application)
    await server.start_server()
    try:
        yield server
    finally:
        await server.close()


def _application(handler: _Handler) -> web.Application:
    application = web.Application()
    application.router.add_route("*", "/{path:.*}", handler)
    return application


def _backend(server: TestServer) -> _Backend:
    assert server.host is not None
    assert server.port is not None
    return _Backend(server.host, server.port)


@pytest.mark.asyncio
async def test_proxy_retries_overload_on_another_least_active_backend() -> None:
    calls: list[str] = []

    async def overloaded(request: web.Request) -> web.Response:
        calls.append("overloaded")
        assert await request.read() == b"{}"
        return web.Response(status=429, text="busy")

    async def available(request: web.Request) -> web.Response:
        calls.append("available")
        assert await request.read() == b"{}"
        return web.json_response({"status": "complete"})

    async with _serve(_application(overloaded)) as first, _serve(_application(available)) as second:
        proxy = _ProxyApplication((_backend(first), _backend(second)), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), data=b"{}") as response:
                assert response.status == 200
                assert await response.json() == {"status": "complete"}

    assert calls == ["overloaded", "available"]


@pytest.mark.asyncio
async def test_proxy_limits_failover_to_three_distinct_backends_and_preserves_final_retry_after() -> None:
    calls: list[int] = []

    def build_handler(index: int) -> _Handler:
        async def overloaded(request: web.Request) -> web.Response:
            del request
            calls.append(index)
            return web.Response(status=429, text=f"busy-{index}", headers={"Retry-After": "99"})

        return overloaded

    applications = tuple(_application(build_handler(index)) for index in range(4))
    async with (
        _serve(applications[0]) as first,
        _serve(applications[1]) as second,
        _serve(applications[2]) as third,
        _serve(applications[3]) as fourth,
    ):
        backends = tuple(_backend(server) for server in (first, second, third, fourth))
        proxy = _ProxyApplication(backends, 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), json={}) as response:
                assert response.status == 429
                assert await response.text() == f"busy-{calls[-1]}"
                assert response.headers["Retry-After"] == "99"

    assert calls == [0, 1, 2]


@pytest.mark.asyncio
async def test_proxy_adds_configured_retry_after_when_final_backend_omits_it() -> None:
    async def overloaded(request: web.Request) -> web.Response:
        del request
        return web.Response(status=429, text="busy")

    async with _serve(_application(overloaded)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 3)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), json={}) as response:
                assert response.status == 429
                assert response.headers["Retry-After"] == "3"


@pytest.mark.asyncio
async def test_proxy_rejects_requests_over_the_configured_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = 0

    async def available(request: web.Request) -> web.Response:
        nonlocal calls
        calls += 1
        return web.Response(text="complete")

    monkeypatch.setattr(runtime_proxy, "_MAXIMUM_REQUEST_BYTES", 4)
    async with _serve(_application(available)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), data=b"12345") as response:
                assert response.status == 413

    assert calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("status", (500, 502, 503, 504))
async def test_proxy_retries_transient_http_failures(status: int) -> None:
    async def unavailable(request: web.Request) -> web.Response:
        del request
        return web.Response(status=status)

    async def available(request: web.Request) -> web.Response:
        del request
        return web.Response(text="complete")

    async with _serve(_application(unavailable)) as first, _serve(_application(available)) as second:
        proxy = _ProxyApplication((_backend(first), _backend(second)), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), json={}) as response:
                assert response.status == 200
                assert await response.text() == "complete"


@pytest.mark.asyncio
async def test_proxy_retries_connection_failure_on_another_backend() -> None:
    unavailable = socket.socket()
    unavailable.bind(("127.0.0.1", 0))
    unavailable_port = unavailable.getsockname()[1]
    unavailable.close()

    async def available(request: web.Request) -> web.Response:
        del request
        return web.Response(text="complete")

    async with _serve(_application(available)) as server:
        proxy = _ProxyApplication((_Backend("127.0.0.1", unavailable_port), _backend(server)), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), json={}) as response:
                assert response.status == 200
                assert await response.text() == "complete"


@pytest.mark.asyncio
async def test_proxy_preserves_overload_after_final_backend_connection_failure() -> None:
    unavailable = socket.socket()
    unavailable.bind(("127.0.0.1", 0))
    unavailable_port = unavailable.getsockname()[1]
    unavailable.close()

    async def overloaded(request: web.Request) -> web.Response:
        del request
        return web.Response(status=429, headers={"Retry-After": "17"})

    async with _serve(_application(overloaded)) as server:
        proxy = _ProxyApplication((_backend(server), _Backend("127.0.0.1", unavailable_port)), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.post(endpoint.make_url("/v1/chat/completions"), json={}) as response:
                assert response.status == 429
                assert response.headers["Retry-After"] == "17"


@pytest.mark.asyncio
async def test_proxy_streams_response_without_waiting_for_completion() -> None:
    release_tail = asyncio.Event()

    async def streaming(request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(status=200)
        await response.prepare(request)
        await response.write(b"first")
        await release_tail.wait()
        await response.write(b"second")
        await response.write_eof()
        return response

    async with _serve(_application(streaming)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.get(endpoint.make_url("/v1/models")) as response:
                assert await asyncio.wait_for(response.content.readexactly(5), timeout=1) == b"first"
                release_tail.set()
                assert await response.read() == b"second"


@pytest.mark.asyncio
async def test_proxy_reuses_upstream_connections_and_reports_pool_metrics() -> None:
    transports: set[object] = set()

    async def available(request: web.Request) -> web.Response:
        assert request.transport is not None
        transports.add(request.transport)
        return web.Response(text="complete")

    async with _serve(_application(available)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            for _ in range(2):
                async with client.get(endpoint.make_url("/v1/models")) as response:
                    assert await response.text() == "complete"
            async with client.get(endpoint.make_url("/metrics")) as response:
                metrics = await response.json()

    assert len(transports) == 1
    assert metrics["connections"]["opened"] == 1
    assert metrics["connections"]["reused"] == 1
    assert metrics["connections"]["idle"] == 1


@pytest.mark.asyncio
async def test_proxy_waits_for_a_pooled_connection_without_timing_out(monkeypatch: pytest.MonkeyPatch) -> None:
    first_started = asyncio.Event()
    release_first = asyncio.Event()
    requests = 0

    async def available(request: web.Request) -> web.Response:
        del request
        nonlocal requests
        requests += 1
        if requests == 1:
            first_started.set()
            await release_first.wait()
        return web.Response(text="complete")

    monkeypatch.setattr(runtime_proxy, "_MAXIMUM_CONNECTIONS", 1)
    monkeypatch.setattr(runtime_proxy, "_CONNECT_TIMEOUT_SECONDS", 0.05)
    async with _serve(_application(available)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            first = asyncio.create_task(client.get(endpoint.make_url("/first")))
            await first_started.wait()
            second = asyncio.create_task(client.get(endpoint.make_url("/second")))
            await asyncio.sleep(0.1)
            release_first.set()
            async with await first as response:
                assert response.status == 200
                assert await response.text() == "complete"
            async with await second as response:
                assert response.status == 200
                assert await response.text() == "complete"

    assert requests == 2


@pytest.mark.asyncio
async def test_proxy_releases_backend_when_upstream_request_is_cancelled(monkeypatch: pytest.MonkeyPatch) -> None:
    proxy = _ProxyApplication((_Backend("127.0.0.1", 8001),), 1)

    class _Request:
        async def read(self) -> bytes:
            return b""

    async def cancelled(*args: object) -> None:
        del args
        raise asyncio.CancelledError

    monkeypatch.setattr(proxy, "_request_backend", cancelled)
    with pytest.raises(asyncio.CancelledError):
        await proxy._forward(_Request())  # type: ignore[arg-type]

    assert proxy.pool.active_counts() == (0,)


@pytest.mark.asyncio
async def test_proxy_health_requires_every_backend_to_be_ready() -> None:
    async def healthy(request: web.Request) -> web.Response:
        return web.Response(status=200 if request.path == "/ready" else 404)

    async def unhealthy(request: web.Request) -> web.Response:
        return web.Response(status=503 if request.path == "/ready" else 404)

    async with _serve(_application(healthy)) as first, _serve(_application(unhealthy)) as second:
        proxy = _ProxyApplication((_backend(first), _backend(second)), 1, health_path="/ready")
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.get(endpoint.make_url("/health")) as response:
                assert response.status == 503
                assert await response.json() == {"status": "unavailable", "backends_ready": 1, "backends": 2}


@pytest.mark.asyncio
async def test_proxy_strips_connection_nominated_headers_in_both_directions() -> None:
    forwarded_headers: dict[str, str] = {}

    async def available(request: web.Request) -> web.Response:
        forwarded_headers.update(request.headers)
        return web.Response(
            text="complete",
            headers={"Connection": "X-Backend-Hop", "X-Backend-Hop": "private"},
        )

    async with _serve(_application(available)) as backend:
        proxy = _ProxyApplication((_backend(backend),), 1)
        async with _serve(proxy.create()) as endpoint, ClientSession() as client:
            async with client.get(
                endpoint.make_url("/v1/models"),
                headers={
                    "Authorization": "Bearer reviewed",
                    "Connection": "X-Client-Hop",
                    "X-Client-Hop": "private",
                },
            ) as response:
                assert await response.text() == "complete"
                assert "X-Backend-Hop" not in response.headers

    assert forwarded_headers["Authorization"] == "Bearer reviewed"
    assert "X-Client-Hop" not in forwarded_headers


@pytest.mark.parametrize(
    "value",
    (
        "https://127.0.0.1:8000",
        "http://example.com:8000",
        "http://127.0.0.1:70000",
        "http://user@127.0.0.1:8000",
        "http://127.0.0.1:8000/path",
    ),
)
def test_proxy_rejects_non_loopback_or_malformed_backend(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="backend"):
        _parse_backend(value)


def test_proxy_matches_allowed_backend_hosts_case_insensitively() -> None:
    backend = _parse_backend("http://compute-001:8000", frozenset({"Compute-001"}))

    assert backend == _Backend("compute-001", 8000)


def test_pool_selects_least_active_backend_deterministically() -> None:
    pool = _BackendPool(
        (_Backend("127.0.0.1", 8001), _Backend("127.0.0.1", 8002), _Backend("127.0.0.1", 8003)),
        1,
    )
    first = pool.acquire(frozenset())
    second = pool.acquire(frozenset())
    pool.release(first)
    third = pool.acquire(frozenset())
    pool.release(second)
    pool.release(third)

    assert (first, second, third) == (0, 1, 2)
    assert pool.active_counts() == (0, 0, 0)
