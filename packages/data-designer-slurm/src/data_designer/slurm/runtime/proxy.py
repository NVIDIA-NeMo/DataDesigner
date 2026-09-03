# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Allocation-local least-connections HTTP endpoint for one model alias."""

from __future__ import annotations

import argparse
import http.client
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import SplitResult, urlsplit

_MAXIMUM_REQUEST_BYTES = 64 * 1024 * 1024
_MAXIMUM_RESPONSE_BYTES = 64 * 1024 * 1024
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


class _BackendPool:
    def __init__(self, backends: tuple[_Backend, ...], retry_after_seconds: int | None) -> None:
        self.backends = backends
        self.retry_after_seconds = retry_after_seconds
        self._active = [0] * len(backends)
        self._cursor = 0
        self._lock = threading.Lock()

    def acquire(self, excluded: frozenset[int]) -> int:
        with self._lock:
            available = tuple(index for index in range(len(self.backends)) if index not in excluded)
            if not available:
                raise LookupError("no backend remains")
            minimum = min(self._active[index] for index in available)
            candidates = tuple(index for index in available if self._active[index] == minimum)
            index = candidates[self._cursor % len(candidates)]
            self._cursor += 1
            self._active[index] += 1
            return index

    def release(self, index: int) -> None:
        with self._lock:
            self._active[index] -= 1


class _ProxyServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address: tuple[str, int], pool: _BackendPool) -> None:
        self.pool = pool
        super().__init__(address, _ProxyHandler)


class _ProxyHandler(BaseHTTPRequestHandler):
    server: _ProxyServer
    protocol_version = "HTTP/1.1"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._write_response(200, b'{"status":"ok"}\n', {"Content-Type": "application/json"})
            return
        self._forward()

    def do_POST(self) -> None:  # noqa: N802
        self._forward()

    def do_DELETE(self) -> None:  # noqa: N802
        self._forward()

    def do_PUT(self) -> None:  # noqa: N802
        self._forward()

    def log_message(self, format: str, *args: object) -> None:
        del format, args

    def _forward(self) -> None:
        if not self.path.startswith("/") or self.path.startswith("//"):
            self.send_error(400)
            return
        request_body = self._read_request_body()
        if request_body is None:
            return
        final_response = self._try_backends(request_body)
        if final_response is None:
            self.send_error(503)
            return
        if final_response[0] == 429:
            final_response = (
                final_response[0],
                final_response[1],
                _retry_after_headers(final_response[2], self.server.pool.retry_after_seconds),
            )
        self._write_response(*final_response)

    def _try_backends(self, request_body: bytes) -> tuple[int, bytes, dict[str, str]] | None:
        excluded: set[int] = set()
        final_response: tuple[int, bytes, dict[str, str]] | None = None
        while len(excluded) < len(self.server.pool.backends):
            index = self.server.pool.acquire(frozenset(excluded))
            try:
                final_response = self._request_backend(self.server.pool.backends[index], request_body)
            finally:
                self.server.pool.release(index)
            if final_response[0] != 429:
                break
            excluded.add(index)
        return final_response

    def _read_request_body(self) -> bytes | None:
        if self.headers.get("Transfer-Encoding") is not None:
            self.send_error(501)
            return None
        lengths = self.headers.get_all("Content-Length", failobj=[])
        if len(lengths) > 1:
            self.send_error(400)
            return None
        raw_length = lengths[0] if lengths else "0"
        try:
            length = int(raw_length)
        except ValueError:
            self.send_error(400)
            return None
        if length < 0 or length > _MAXIMUM_REQUEST_BYTES:
            self.send_error(413)
            return None
        content = self.rfile.read(length)
        if len(content) != length:
            self.send_error(400)
            return None
        return content

    def _request_backend(self, backend: _Backend, body: bytes) -> tuple[int, bytes, dict[str, str]]:
        request_headers = tuple(self.headers.items())
        hop_headers = _get_hop_headers(request_headers)
        headers = {
            name: value
            for name, value in request_headers
            if name.lower() not in hop_headers and name.lower() not in {"host", "content-length"}
        }
        headers["Content-Length"] = str(len(body))
        connection = http.client.HTTPConnection(backend.host, backend.port, timeout=300)
        try:
            connection.request(self.command, self.path, body=body, headers=headers)
            response = connection.getresponse()
            payload = response.read(_MAXIMUM_RESPONSE_BYTES + 1)
            if len(payload) > _MAXIMUM_RESPONSE_BYTES:
                return 502, b'{"error":"backend response too large"}\n', {"Content-Type": "application/json"}
            raw_response_headers = tuple(response.getheaders())
            response_hop_headers = _get_hop_headers(raw_response_headers)
            response_headers = {
                name: value
                for name, value in raw_response_headers
                if name.lower() not in response_hop_headers and name.lower() != "content-length"
            }
            return response.status, payload, response_headers
        except (OSError, http.client.HTTPException):
            return 502, b'{"error":"backend unavailable"}\n', {"Content-Type": "application/json"}
        finally:
            connection.close()

    def _write_response(self, status: int, body: bytes, headers: dict[str, str]) -> None:
        self.send_response(status)
        for name, value in headers.items():
            self.send_header(name, value)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)


def main(arguments: Sequence[str] | None = None) -> int:
    """Serve one loopback logical endpoint until the step is terminated."""
    parser = argparse.ArgumentParser(prog="data-designer-slurm-proxy")
    parser.add_argument("--listen-port", required=True, type=int)
    parser.add_argument("--backend", action="append", required=True)
    parser.add_argument("--retry-after-seconds", type=int)
    parsed = parser.parse_args(arguments)
    backends = tuple(_parse_backend(value) for value in parsed.backend)
    if not 1 <= parsed.listen_port <= 65535:
        parser.error("listen port must be between 1 and 65535")
    if parsed.retry_after_seconds is not None and parsed.retry_after_seconds <= 0:
        parser.error("retry-after seconds must be positive")
    pool = _BackendPool(backends, parsed.retry_after_seconds)
    with _ProxyServer(("127.0.0.1", parsed.listen_port), pool) as server:
        server.serve_forever(poll_interval=0.1)
    return 0


def _parse_backend(value: str) -> _Backend:
    parsed: SplitResult = urlsplit(value)
    try:
        port = parsed.port
    except ValueError as error:
        raise argparse.ArgumentTypeError("backend port is invalid") from error
    if (
        parsed.scheme != "http"
        or parsed.hostname != "127.0.0.1"
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or port is None
    ):
        raise argparse.ArgumentTypeError("backends must be loopback HTTP origins")
    return _Backend(parsed.hostname, port)


def _retry_after_headers(headers: dict[str, str], retry_after_seconds: int | None) -> dict[str, str]:
    selected = {name: value for name, value in headers.items() if name.lower() != "retry-after"}
    if retry_after_seconds is not None:
        selected["Retry-After"] = str(retry_after_seconds)
    return selected


def _get_hop_headers(headers: Sequence[tuple[str, str]]) -> frozenset[str]:
    nominated = {
        token.strip().lower()
        for name, value in headers
        if name.lower() == "connection"
        for token in value.split(",")
        if token.strip()
    }
    return _HOP_HEADERS | nominated


if __name__ == "__main__":  # pragma: no cover - exercised as a managed step
    raise SystemExit(main())
