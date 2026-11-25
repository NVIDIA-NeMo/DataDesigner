# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import errno
import os
import resource
import signal
import socket
import sys
import threading
from contextlib import AsyncExitStack
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp import ClientConnectorError, ClientError, ClientResponse, ClientSession, TCPConnector, web
from aiohttp.test_utils import TestServer

from data_designer.slurm.runtime import proxy as runtime_proxy
from data_designer.slurm.runtime.proxy import _Backend, _ProxyApplication, _SocketBudget, _SocketBudgetExceeded


@pytest.mark.parametrize(
    ("soft_limit", "backend_count"),
    [(128, 1), (1024, 4), (131072, 4), (1024, 100)],
)
def test_budget_uses_effective_soft_limit_and_reserves_descriptors(
    monkeypatch: pytest.MonkeyPatch, soft_limit: int, backend_count: int
) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (soft_limit, 500000))
    monkeypatch.setattr(_SocketBudget, "count_open", lambda _: 16)
    budget = _SocketBudget(backend_count)
    assert budget.soft_limit == soft_limit
    assert budget.headroom >= backend_count + 32
    assert budget.connection_limit > 0
    assert 2 * budget.connection_limit + budget.headroom + 16 <= soft_limit
    if soft_limit >= 1024:
        assert budget.connection_limit > 256


@pytest.mark.parametrize("soft_limit", [32, 64, 80, 81])
def test_startup_rejects_descriptor_limits_without_inference_capacity(
    monkeypatch: pytest.MonkeyPatch, soft_limit: int
) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (soft_limit, soft_limit))
    monkeypatch.setattr(_SocketBudget, "count_open", lambda _: 16)
    with pytest.raises(RuntimeError, match=rf"RLIMIT_NOFILE.*soft={soft_limit}.*open=16.*reserved=64.*increase"):
        _SocketBudget(1)


def test_unlimited_soft_limit_uses_system_open_max(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    monkeypatch.setattr(os, "sysconf", lambda _: 65536)
    monkeypatch.setattr(_SocketBudget, "count_open", lambda _: 16)
    assert _SocketBudget(1).soft_limit == 65536
    monkeypatch.setattr(os, "sysconf", lambda _: -1)
    with pytest.raises(RuntimeError, match="cannot determine"):
        _SocketBudget(1)


@pytest.mark.asyncio
async def test_live_descriptor_pressure_blocks_socket_creation_and_recovers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (1024, 1024))
    monkeypatch.setattr(_SocketBudget, "count_open", lambda _: 16)
    budget = _SocketBudget(4)
    factory = Mock(wraps=runtime_proxy._create_socket)
    monkeypatch.setattr(runtime_proxy, "_create_socket", factory)
    address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 8000))
    monkeypatch.setattr(budget, "count_open", lambda: 896)
    await budget.refresh()
    assert not budget.acquire()
    with pytest.raises(_SocketBudgetExceeded):
        budget.create_socket(address)
    factory.assert_not_called()
    assert budget.in_flight == 0
    monkeypatch.setattr(budget, "count_open", lambda: 895)
    await budget.refresh()
    assert budget.acquire()
    with budget.create_socket(address):
        assert not budget.has_headroom()
    assert budget.has_headroom()
    budget.release()
    assert budget.in_flight == 0


def test_budget_rejects_without_queueing_at_its_in_flight_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(resource, "getrlimit", lambda _: (128, 128))
    monkeypatch.setattr(_SocketBudget, "count_open", lambda _: 16)
    budget = _SocketBudget(1)
    assert all(budget.acquire() for _ in range(budget.connection_limit))
    assert not budget.acquire()
    budget.release()
    assert budget.acquire()


@pytest.mark.asyncio
async def test_descriptor_inspection_failure_preserves_headroom(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = _SocketBudget(1)
    monkeypatch.setattr(budget, "count_open", Mock(side_effect=OSError("descriptor table unavailable")))
    await budget.refresh()
    assert not budget.acquire()


@pytest.mark.asyncio
async def test_pressure_discards_idle_upstream_connections_and_closes_overload_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def respond(request: web.Request) -> web.Response:
        return web.Response(text="complete")

    app = web.Application()
    app.router.add_get("/{path:.*}", respond)
    async with TestServer(app) as backend:
        proxy = _ProxyApplication((_Backend(backend.host, backend.port),), 3)
        async with TestServer(proxy.create()) as endpoint, ClientSession() as client:
            async with client.get(endpoint.make_url("/complete")) as response:
                assert response.status == 200
                await response.read()
            assert proxy.connector is not None
            assert proxy.budget is not None
            assert sum(len(items) for items in proxy.connector._conns.values()) == 1
            original_count = proxy.budget.count_open
            monkeypatch.setattr(proxy.budget, "count_open", lambda: proxy.budget.soft_limit)
            await proxy.budget.refresh()
            async with client.get(endpoint.make_url("/complete")) as response:
                assert response.status == 429
                assert response.headers["Retry-After"] == "3"
                assert response.headers["Connection"] == "close"
                await response.read()
            assert not proxy.connector._conns
            assert proxy.budget.in_flight == 0
            monkeypatch.setattr(proxy.budget, "count_open", original_count)
            await proxy.budget.refresh()
            async with client.get(endpoint.make_url("/complete")) as response:
                assert response.status == 200
                await response.read()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "error_code", [None, errno.EMFILE, errno.ENFILE], ids=["budget", "process-limit", "system-limit"]
)
async def test_socket_pressure_returns_429_without_retrying_backends(
    monkeypatch: pytest.MonkeyPatch, error_code: int | None
) -> None:
    error = (
        _SocketBudgetExceeded("socket budget")
        if error_code is None
        else ClientConnectorError(Mock(), OSError(error_code, "descriptor exhaustion"))
    )
    proxy = _ProxyApplication((_Backend("127.0.0.1", 8001), _Backend("127.0.0.1", 8002)), None)
    request_backend = AsyncMock(side_effect=error)
    monkeypatch.setattr(proxy, "_request_backend", request_backend)
    async with TestServer(proxy.create()) as endpoint, ClientSession() as client:
        async with client.get(endpoint.make_url("/complete")) as response:
            assert response.status == 429
            assert response.headers["Retry-After"] == "1"
            assert response.headers["Connection"] == "close"
            await response.read()
        assert request_backend.await_count == 1
        assert proxy.pool.active_counts() == (0, 0)
        assert proxy.budget is not None and proxy.budget.in_flight == 0


@pytest.mark.asyncio
async def test_real_proxy_preserves_health_and_returns_overload_under_1024_descriptor_limit() -> None:
    parent_budget = _SocketBudget(4)
    if parent_budget.soft_limit - parent_budget.count_open() < 2200:
        pytest.skip("parent HTTP fixture needs 2200 free descriptors; proxy itself is tested with a limit of 1024")
    _, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < 1024:
        pytest.skip("subprocess cannot set a 1024-descriptor soft limit under the host hard limit")
    release = asyncio.Event()
    backend_active = 0

    async def respond(request: web.Request) -> web.Response:
        nonlocal backend_active
        if request.path == "/health":
            return web.Response()
        backend_active += 1
        try:
            await release.wait()
            return web.Response(text="complete")
        finally:
            backend_active -= 1

    async with AsyncExitStack() as stack:
        # Distinct backend origins also exercise pooled connections across hosts.
        servers = []
        for _ in range(4):
            backend_app = web.Application()
            backend_app.router.add_get("/{path:.*}", respond)
            servers.append(await stack.enter_async_context(TestServer(backend_app)))
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        command = (
            "import resource, sys; "
            "resource.setrlimit(resource.RLIMIT_NOFILE, (1024, resource.getrlimit(resource.RLIMIT_NOFILE)[1])); "
            "from data_designer.slurm.runtime.proxy import main; sys.exit(main())"
        )
        args = [sys.executable, "-c", command, "--listen-port", str(port)]
        for backend in servers:
            args.extend(["--backend", f"http://{backend.host}:{backend.port}"])
        process = await asyncio.create_subprocess_exec(
            *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
        )
        output = asyncio.create_task(process.communicate())
        client = await stack.enter_async_context(ClientSession(connector=TCPConnector(limit=0, force_close=True)))
        origin = f"http://127.0.0.1:{port}"
        requests: list[asyncio.Task[ClientResponse]] = []
        try:
            async with asyncio.timeout(10):
                while True:
                    try:
                        async with client.get(origin + "/health") as response:
                            assert response.status == 200
                            break
                    except ClientError:
                        assert process.returncode is None
                        await asyncio.sleep(0.02)
            requests = [asyncio.create_task(client.get(origin + "/complete")) for _ in range(520)]
            async with asyncio.timeout(15):
                while backend_active + sum(task.done() for task in requests) < 520:
                    await asyncio.sleep(0.02)
            rejected = [task.result() for task in requests if task.done()]
            assert rejected
            assert all(response.status == 429 and response.headers["Retry-After"] == "1" for response in rejected)
            async with client.get(origin + "/health") as response:
                assert response.status == 200
                assert (await response.json())["backends_ready"] == 4
            async with client.get(origin + "/metrics") as response:
                metrics = await response.json()
            assert metrics["socket_budget"]["soft_limit"] == 1024
            assert 256 < metrics["connections"]["limit"] < 512
            assert 0 < backend_active <= metrics["connections"]["limit"]
            release.set()
            responses = await asyncio.wait_for(asyncio.gather(*requests), 10)
            for response in responses:
                assert response.status in {200, 429}
                await response.read()
                response.release()
            for _ in rejected:
                async with client.get(origin + "/complete") as response:
                    assert response.status == 200
                    await response.read()
        finally:
            release.set()
            for task in requests:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*requests, return_exceptions=True)
            if process.returncode is None:
                process.terminate()
            _, stderr = await asyncio.wait_for(output, 10)
        assert b"Too many open files" not in stderr
        assert b"EMFILE" not in stderr


@pytest.mark.asyncio
async def test_admission_budget_returns_immediate_overload_and_releases_capacity() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(request: web.Request) -> web.Response:
        if request.path == "/health":
            return web.Response()
        entered.set()
        await release.wait()
        return web.Response(text="complete")

    app = web.Application()
    app.router.add_get("/{path:.*}", respond)
    async with TestServer(app) as backend:
        proxy = _ProxyApplication((_Backend(backend.host, backend.port),), 1)
        async with TestServer(proxy.create()) as endpoint, ClientSession() as client:
            assert proxy.budget is not None
            proxy.budget.connection_limit = 1
            first = asyncio.create_task(client.get(endpoint.make_url("/complete")))
            try:
                await asyncio.wait_for(entered.wait(), 2)
                async with asyncio.timeout(2):
                    async with client.get(endpoint.make_url("/complete")) as response:
                        assert response.status == 429
                        await response.read()
                    async with client.get(endpoint.make_url("/health")) as response:
                        assert response.status == 200
                        await response.read()
                assert proxy.budget.in_flight == 1
            finally:
                release.set()
                async with await first as response:
                    assert response.status == 200
                    await response.read()
            async with client.get(endpoint.make_url("/complete")) as response:
                assert response.status == 200
                await response.read()
            assert proxy.budget.in_flight == 0


def test_admission_and_socket_lifecycle_do_not_scan_descriptors(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = _SocketBudget(1)
    scan = Mock(side_effect=AssertionError("request paths must not scan descriptors"))
    monkeypatch.setattr(budget, "count_open", scan)
    before = budget._estimated_open
    address = (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 8000))
    assert budget.acquire()
    connection = budget.create_socket(address)
    assert budget._estimated_open == before + 1
    connection.close()
    connection.close()
    budget.release()
    assert budget._estimated_open == before
    scan.assert_not_called()


@pytest.mark.asyncio
async def test_reconciliation_preserves_opens_during_scan_without_crediting_racing_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    budget = _SocketBudget(1)
    started, release = threading.Event(), threading.Event()

    def scan() -> int:
        started.set()
        assert release.wait(2)
        return 100

    monkeypatch.setattr(budget, "count_open", scan)
    refresh = asyncio.create_task(budget.refresh())
    try:
        assert await asyncio.to_thread(started.wait, 2)
        for _ in range(3):
            budget.connection_opened()
        for _ in range(2):
            budget.connection_closed()
    finally:
        release.set()
        await refresh
    assert budget._estimated_open == 103
    budget.connection_closed()
    assert budget._estimated_open == 102


@pytest.mark.asyncio
async def test_stale_sample_blocks_admission_until_reconciliation(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = _SocketBudget(1)
    budget._sampled_at -= runtime_proxy._DESCRIPTOR_SAMPLE_MAXIMUM_AGE_SECONDS + 1
    assert not budget.acquire()
    monkeypatch.setattr(budget, "count_open", lambda: 16)
    await budget.refresh()
    assert budget.acquire()
    budget.release()


@pytest.mark.asyncio
async def test_sampler_runs_off_event_loop_and_stops_with_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    budget = _SocketBudget(1)
    loop_thread = threading.get_ident()
    threads: list[int] = []

    def scan() -> int:
        threads.append(threading.get_ident())
        return 16

    monkeypatch.setattr(budget, "count_open", scan)
    monkeypatch.setattr(runtime_proxy, "_DESCRIPTOR_SAMPLE_INTERVAL_SECONDS", 0.001)
    budget.start()
    try:
        async with asyncio.timeout(2):
            while not threads:
                await asyncio.sleep(0.001)
    finally:
        await budget.close()
    assert threads and all(thread != loop_thread for thread in threads)
    assert budget._monitor_task is not None and budget._monitor_task.done()


@pytest.mark.asyncio
async def test_production_server_tracks_connections_before_http_request() -> None:
    proxy = _ProxyApplication((_Backend("127.0.0.1", 8001),), 1)
    runner = runtime_proxy._ProxyRunner(proxy)
    await runner.setup()
    try:
        await web.TCPSite(runner, "127.0.0.1", 0).start()
        assert proxy.budget is not None
        await proxy.budget.close()
        before = proxy.budget._estimated_open
        _, writer = await asyncio.open_connection("127.0.0.1", runner.addresses[0][1])
        try:
            async with asyncio.timeout(2):
                while proxy.budget._estimated_open == before:
                    await asyncio.sleep(0.001)
            assert proxy.budget._estimated_open == before + 1
        finally:
            writer.close()
            await writer.wait_closed()
        async with asyncio.timeout(2):
            while proxy.budget._estimated_open != before:
                await asyncio.sleep(0.001)
    finally:
        await runner.cleanup()


@pytest.mark.asyncio
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT], ids=["sigterm", "sigint"])
@pytest.mark.parametrize("streaming", [False, True], ids=["before-headers", "streaming"])
async def test_real_signal_drains_active_response_before_process_exit(
    stop_signal: signal.Signals, streaming: bool
) -> None:
    entered, first_received, release = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def respond(request: web.Request) -> web.StreamResponse:
        if request.path == "/health":
            return web.Response()
        if streaming:
            response = web.StreamResponse()
            await response.prepare(request)
            await response.write(b"first")
            entered.set()
            await release.wait()
            await response.write(b"second")
            return response
        entered.set()
        await release.wait()
        return web.Response(body=b"complete")

    app = web.Application()
    app.router.add_route("*", "/{path:.*}", respond)
    async with TestServer(app) as backend, ClientSession() as client:
        with socket.socket() as reservation:
            reservation.bind(("127.0.0.1", 0))
            port = reservation.getsockname()[1]
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "data_designer.slurm.runtime.proxy",
            "--listen-port",
            str(port),
            "--backend",
            f"http://{backend.host}:{backend.port}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        output = asyncio.create_task(process.communicate())
        origin = f"http://127.0.0.1:{port}"
        request_task: asyncio.Task[bytes] | None = None

        async def read_response() -> bytes:
            async with client.post(origin + "/complete") as response:
                assert response.status == 200
                first = await response.content.readexactly(5) if streaming else b""
                first_received.set()
                return first + await response.read()

        try:
            async with asyncio.timeout(10):
                while True:
                    try:
                        async with client.get(origin + "/health") as response:
                            assert response.status == 200
                            await response.read()
                            break
                    except ClientError:
                        assert process.returncode is None
                        await asyncio.sleep(0.02)
            request_task = asyncio.create_task(read_response())
            await asyncio.wait_for(entered.wait(), 2)
            if streaming:
                await asyncio.wait_for(first_received.wait(), 2)
            process.send_signal(stop_signal)
            await asyncio.sleep(0.1)
            assert not request_task.done(), "signal cut off the active response before the shutdown window"
            assert process.returncode is None
            release.set()
            assert await asyncio.wait_for(request_task, 3) == (b"firstsecond" if streaming else b"complete")
            await asyncio.wait_for(output, 5)
            assert process.returncode == 0
        finally:
            release.set()
            if request_task is not None:
                if not request_task.done():
                    request_task.cancel()
                await asyncio.gather(request_task, return_exceptions=True)
            if process.returncode is None:
                process.terminate()
            await asyncio.wait_for(output, 12)


@pytest.mark.asyncio
async def test_real_proxy_exits_with_diagnostic_when_descriptor_limit_is_64(tmp_path: Path) -> None:
    marker = tmp_path / "proxy.listening"
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        port = reservation.getsockname()[1]
    command = (
        "import resource, sys; "
        "resource.setrlimit(resource.RLIMIT_NOFILE, (64, resource.getrlimit(resource.RLIMIT_NOFILE)[1])); "
        "from data_designer.slurm.runtime.proxy import main; sys.exit(main())"
    )
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        command,
        "--listen-port",
        str(port),
        "--backend",
        "http://127.0.0.1:8001",
        "--listening-file",
        str(marker),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    output = asyncio.create_task(process.communicate())
    try:
        _, stderr = await asyncio.wait_for(asyncio.shield(output), 5)
        assert process.returncode != 0
        message = stderr.decode()
        assert "proxy cannot forward inference with RLIMIT_NOFILE soft=64" in message
        assert "open=" in message and "reserved=64" in message
        assert "increase the process file-descriptor limit" in message
        assert not marker.exists()
    finally:
        if process.returncode is None:
            process.terminate()
        await asyncio.wait_for(output, 12)
