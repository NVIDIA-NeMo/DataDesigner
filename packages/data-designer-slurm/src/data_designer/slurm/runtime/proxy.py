# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local streaming HTTP endpoint for one model alias."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import socket
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import SplitResult, urlsplit

from aiohttp import ClientError, ClientResponse, ClientSession, ClientTimeout, TCPConnector, TraceConfig, web

from data_designer.slurm.runtime.network import validate_host_name

_LOGGER = logging.getLogger("data_designer.slurm.runtime.proxy")
_MAXIMUM_REQUEST_BYTES = 100 * 1024 * 1024
_MAXIMUM_UPSTREAM_ATTEMPTS = 3
_MAXIMUM_CONNECTIONS = 256
_KEEPALIVE_TIMEOUT_SECONDS = 65.0
_CONNECT_TIMEOUT_SECONDS = 5.0
_READ_TIMEOUT_SECONDS = 3600.0
_STREAM_CHUNK_BYTES = 64 * 1024
_RETRYABLE_STATUS_CODES = frozenset({429, 500, 502, 503, 504})
_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


@dataclass(frozen=True, slots=True)
class _Backend:
    host: str
    port: int

    @property
    def origin(self) -> str:
        return f"http://{self.host}:{self.port}"


class _BackendPool:
    """Track active requests for deterministic least-connections routing."""

    def __init__(self, backends: tuple[_Backend, ...], retry_after_seconds: int | None) -> None:
        if not backends:
            raise ValueError("at least one backend is required")
        self.backends = backends
        self.retry_after_seconds = retry_after_seconds
        self._active = [0] * len(backends)
        self._cursor = 0

    def acquire(self, excluded: frozenset[int]) -> int:
        available = tuple(index for index in range(len(self.backends)) if index not in excluded)
        if not available:
            raise LookupError("no backend remains")
        minimum = min(self._active[index] for index in available)
        candidates = frozenset(index for index in available if self._active[index] == minimum)
        index = next(
            candidate
            for offset in range(len(self.backends))
            if (candidate := (self._cursor + offset) % len(self.backends)) in candidates
        )
        self._cursor = (index + 1) % len(self.backends)
        self._active[index] += 1
        return index

    def acquire_index(self, index: int) -> None:
        self._active[index] += 1

    def release(self, index: int) -> None:
        if self._active[index] <= 0:
            raise RuntimeError("backend activity underflow")
        self._active[index] -= 1

    def active_counts(self) -> tuple[int, ...]:
        return tuple(self._active)


class _ProxyMetrics:
    """Allocation-local counters that never contain request content or secrets."""

    def __init__(self) -> None:
        self.requests = 0
        self.attempts = 0
        self.connection_opened = 0
        self.connection_reused = 0
        self.connection_failed = 0
        self.retries: Counter[str] = Counter()
        self.final_statuses: Counter[int] = Counter()
        self.backend_requests: Counter[int] = Counter()

    def snapshot(self, pool: _BackendPool, connector: TCPConnector) -> dict[str, object]:
        idle_connections = sum(len(connections) for connections in connector._conns.values())  # noqa: SLF001
        return {
            "requests": self.requests,
            "attempts": self.attempts,
            "connections": {
                "limit": _MAXIMUM_CONNECTIONS,
                "opened": self.connection_opened,
                "reused": self.connection_reused,
                "failed": self.connection_failed,
                "idle": idle_connections,
            },
            "retries": dict(sorted(self.retries.items())),
            "final_statuses": {str(key): value for key, value in sorted(self.final_statuses.items())},
            "backends": [
                {
                    "index": index,
                    "active_requests": active,
                    "requests": self.backend_requests[index],
                }
                for index, active in enumerate(pool.active_counts())
            ],
        }


class _ProxyApplication:
    """One logical endpoint with a shared upstream connection pool."""

    def __init__(
        self,
        backends: tuple[_Backend, ...],
        retry_after_seconds: int | None,
        *,
        health_path: str = "/health",
    ) -> None:
        self.pool = _BackendPool(backends, retry_after_seconds)
        self.health_path = health_path
        self.metrics = _ProxyMetrics()
        self.connector: TCPConnector | None = None
        self.session: ClientSession | None = None

    def create(self) -> web.Application:
        application = web.Application(client_max_size=_MAXIMUM_REQUEST_BYTES)
        application.on_startup.append(self._start)
        application.on_cleanup.append(self._stop)
        application.router.add_route("GET", "/health", self._health)
        application.router.add_route("GET", "/metrics", self._metrics)
        application.router.add_route("*", "/{path:.*}", self._forward)
        return application

    async def _start(self, application: web.Application) -> None:
        del application
        trace = self._trace_config()
        per_backend_limit = max(1, _MAXIMUM_CONNECTIONS // len(self.pool.backends))
        self.connector = TCPConnector(
            limit=_MAXIMUM_CONNECTIONS,
            limit_per_host=per_backend_limit,
            keepalive_timeout=_KEEPALIVE_TIMEOUT_SECONDS,
            force_close=False,
            socket_factory=_create_socket,
        )
        self.session = ClientSession(
            connector=self.connector,
            timeout=ClientTimeout(
                total=_READ_TIMEOUT_SECONDS,
                sock_connect=_CONNECT_TIMEOUT_SECONDS,
                sock_read=_READ_TIMEOUT_SECONDS,
            ),
            auto_decompress=False,
            raise_for_status=False,
            trace_configs=[trace],
        )

    async def _stop(self, application: web.Application) -> None:
        del application
        if self.session is not None:
            await self.session.close()

    def _trace_config(self) -> TraceConfig:
        trace = TraceConfig()

        async def connection_opened(*_: object) -> None:
            self.metrics.connection_opened += 1

        async def connection_reused(*_: object) -> None:
            self.metrics.connection_reused += 1

        trace.on_connection_create_end.append(connection_opened)
        trace.on_connection_reuseconn.append(connection_reused)
        return trace

    async def _health(self, request: web.Request) -> web.Response:
        del request
        results = await asyncio.gather(
            *(self._probe_backend(index, backend) for index, backend in enumerate(self.pool.backends))
        )
        status = 200 if all(results) else 503
        return web.json_response(
            {
                "status": "ok" if status == 200 else "unavailable",
                "backends_ready": sum(results),
                "backends": len(results),
            },
            status=status,
        )

    async def _probe_backend(self, index: int, backend: _Backend) -> bool:
        session = self._require_session()
        self.pool.acquire_index(index)
        try:
            async with session.get(
                f"{backend.origin}{self.health_path}",
                timeout=ClientTimeout(total=_CONNECT_TIMEOUT_SECONDS),
                trace_request_ctx={"backend_index": index},
            ) as response:
                await response.read()
                return response.status == 200
        except (ClientError, TimeoutError):
            return False
        finally:
            self.pool.release(index)

    async def _metrics(self, request: web.Request) -> web.Response:
        del request
        connector = self._require_connector()
        return web.json_response(self.metrics.snapshot(self.pool, connector))

    async def _forward(self, request: web.Request) -> web.StreamResponse:
        started_at = time.monotonic()
        self.metrics.requests += 1
        body = await request.read()
        excluded: set[int] = set()
        outcomes: list[str] = []
        attempt_limit = min(_MAXIMUM_UPSTREAM_ATTEMPTS, len(self.pool.backends))
        final_response: ClientResponse | None = None
        final_index: int | None = None
        overload_seen = False
        overload_retry_after: str | None = None

        for attempt in range(attempt_limit):
            index = self.pool.acquire(frozenset(excluded))
            backend = self.pool.backends[index]
            self.metrics.attempts += 1
            self.metrics.backend_requests[index] += 1
            try:
                response = await self._request_backend(request, backend, body, index)
            except (ClientError, TimeoutError) as error:
                self.metrics.connection_failed += 1
                reason = "timeout" if isinstance(error, TimeoutError) else "connection_error"
                outcomes.append(reason)
                self.pool.release(index)
                excluded.add(index)
                if attempt + 1 < attempt_limit:
                    self.metrics.retries[reason] += 1
                    continue
                if overload_seen:
                    headers = {}
                    retry_after = (
                        overload_retry_after if overload_retry_after is not None else self.pool.retry_after_seconds
                    )
                    if retry_after is not None:
                        headers["Retry-After"] = str(retry_after)
                    result = web.json_response({"error": "backend overloaded"}, status=429, headers=headers)
                    self._record_final(429, outcomes, started_at)
                    return result
                result = web.json_response({"error": "backend unavailable"}, status=502)
                self._record_final(502, outcomes, started_at)
                return result
            except BaseException:
                self.pool.release(index)
                raise

            outcomes.append(str(response.status))
            should_retry = response.status in _RETRYABLE_STATUS_CODES and attempt + 1 < attempt_limit
            if should_retry:
                reason = f"http_{response.status}"
                self.metrics.retries[reason] += 1
                if response.status == 429:
                    overload_seen = True
                    overload_retry_after = response.headers.get("Retry-After")
                await _discard_response(response)
                self.pool.release(index)
                excluded.add(index)
                continue
            final_response = response
            final_index = index
            break

        if (
            final_response is None or final_index is None
        ):  # pragma: no cover - loop always returns or selects a response
            raise RuntimeError("proxy attempt loop produced no final response")
        try:
            result = await self._stream_response(request, final_response)
            self._record_final(final_response.status, outcomes, started_at)
            return result
        finally:
            final_response.release()
            self.pool.release(final_index)

    async def _request_backend(
        self,
        request: web.Request,
        backend: _Backend,
        body: bytes,
        index: int,
    ) -> ClientResponse:
        session = self._require_session()
        headers = _filter_request_headers(request.headers)
        return await session.request(
            request.method,
            f"{backend.origin}{request.rel_url}",
            data=body,
            headers=headers,
            allow_redirects=False,
            trace_request_ctx={"backend_index": index},
        )

    async def _stream_response(self, request: web.Request, response: ClientResponse) -> web.StreamResponse:
        headers = _filter_response_headers(response.headers)
        has_retry_after = any(name.casefold() == "retry-after" for name in headers)
        if response.status == 429 and not has_retry_after and self.pool.retry_after_seconds is not None:
            headers["Retry-After"] = str(self.pool.retry_after_seconds)
        downstream = web.StreamResponse(status=response.status, reason=response.reason, headers=headers)
        await downstream.prepare(request)
        if request.method != "HEAD":
            async for chunk in response.content.iter_chunked(_STREAM_CHUNK_BYTES):
                await downstream.write(chunk)
        await downstream.write_eof()
        return downstream

    def _record_final(self, status: int, outcomes: list[str], started_at: float) -> None:
        self.metrics.final_statuses[status] += 1
        _LOGGER.info(
            "slurm_proxy_request %s",
            json.dumps(
                {
                    "attempts": len(outcomes),
                    "final_status": status,
                    "upstream_outcomes": outcomes,
                    "elapsed_seconds": round(time.monotonic() - started_at, 6),
                    "failover": len(outcomes) > 1,
                },
                separators=(",", ":"),
                sort_keys=True,
            ),
        )

    def _require_session(self) -> ClientSession:
        if self.session is None:
            raise RuntimeError("proxy session is not started")
        return self.session

    def _require_connector(self) -> TCPConnector:
        if self.connector is None:
            raise RuntimeError("proxy connector is not started")
        return self.connector


async def _discard_response(response: ClientResponse) -> None:
    response.close()


def _filter_request_headers(headers: Mapping[str, str]) -> dict[str, str]:
    hop_headers = _get_hop_headers(tuple(headers.items()))
    return {
        name: value
        for name, value in headers.items()
        if name.casefold() not in hop_headers and name.casefold() not in {"host", "content-length"}
    }


def _filter_response_headers(headers: Mapping[str, str]) -> dict[str, str]:
    hop_headers = _get_hop_headers(tuple(headers.items()))
    return {name: value for name, value in headers.items() if name.casefold() not in hop_headers}


def _create_socket(
    address_info: tuple[int, int, int, str, tuple[str, int] | tuple[str, int, int, int]],
) -> socket.socket:
    family, socket_type, protocol, _, _ = address_info
    connection = socket.socket(family=family, type=socket_type, proto=protocol)
    connection.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return connection


def main(arguments: Sequence[str] | None = None) -> int:
    """Serve one loopback logical endpoint until the step is terminated."""
    parser = argparse.ArgumentParser(prog="data-designer-slurm-proxy")
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--allowed-host", action="append")
    parser.add_argument("--retry-after-seconds", type=int)
    parser.add_argument("--health-path", default="/health")
    parsed = parser.parse_args(arguments)
    try:
        allowed_hosts = frozenset(validate_host_name(value) for value in (parsed.allowed_host or ("127.0.0.1",)))
    except ValueError as error:
        parser.error(str(error))
    backends = tuple(_parse_backend(value, allowed_hosts) for value in parsed.backend)
    if not 1 <= parsed.listen_port <= 65535:
        parser.error("listen port must be between 1 and 65535")
    if parsed.retry_after_seconds is not None and parsed.retry_after_seconds <= 0:
        parser.error("retry-after seconds must be positive")
    if not parsed.health_path.startswith("/") or parsed.health_path.startswith("//"):
        parser.error("health path must be absolute")
    proxy = _ProxyApplication(backends, parsed.retry_after_seconds, health_path=parsed.health_path)
    web.run_app(
        proxy.create(),
        host="127.0.0.1",
        port=parsed.listen_port,
        print=None,
        access_log=None,
        keepalive_timeout=_KEEPALIVE_TIMEOUT_SECONDS,
        shutdown_timeout=10.0,
    )
    return 0


def _parse_backend(value: str, allowed_hosts: frozenset[str] = frozenset({"127.0.0.1"})) -> _Backend:
    parsed: SplitResult = urlsplit(value)
    host = parsed.hostname
    normalized_allowed_hosts = frozenset(allowed_host.casefold() for allowed_host in allowed_hosts)
    try:
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("backend port is invalid") from error
    if (
        parsed.scheme != "http"
        or host is None
        or host.casefold() not in normalized_allowed_hosts
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port is None
    ):
        raise argparse.ArgumentTypeError("backends must be allowed HTTP origins")
    return _Backend(host, port)


def _get_hop_headers(headers: Sequence[tuple[str, str]]) -> frozenset[str]:
    nominated = {
        token.strip().casefold()
        for name, value in headers
        if name.casefold() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    return _HOP_HEADERS | nominated


if __name__ == "__main__":  # pragma: no cover - exercised as a managed step
    raise SystemExit(main())
