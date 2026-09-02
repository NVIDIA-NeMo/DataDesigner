# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import io
from email.message import Message
from types import MethodType
from unittest.mock import patch

import pytest

from data_designer.slurm.runtime import proxy as runtime_proxy
from data_designer.slurm.runtime.proxy import _Backend, _BackendPool, _parse_backend, _ProxyHandler


def test_proxy_retries_overload_on_another_least_active_backend() -> None:
    handler = object.__new__(_ProxyHandler)
    pool = _BackendPool((_Backend("127.0.0.1", 8001), _Backend("127.0.0.1", 8002)), 4, 1)
    handler.server = type("Server", (), {"pool": pool})()
    handler.path = "/v1/chat/completions"
    handler.command = "POST"
    handler.rfile = io.BytesIO(b"{}")
    handler.headers = Message()
    handler.headers["Content-Length"] = "2"
    calls: list[int] = []
    captured: list[tuple[int, bytes, dict[str, str]]] = []

    def request_backend(self: _ProxyHandler, backend: _Backend, body: bytes) -> tuple[int, bytes, dict[str, str]]:
        del self
        calls.append(backend.port)
        assert body == b"{}"
        if backend.port == 8001:
            return 429, b"overloaded", {}
        return 200, b"complete", {"Content-Type": "application/json"}

    def write_response(self: _ProxyHandler, status: int, body: bytes, headers: dict[str, str]) -> None:
        del self
        captured.append((status, body, headers))

    handler._request_backend = MethodType(request_backend, handler)
    handler._write_response = MethodType(write_response, handler)

    handler._forward()

    assert calls == [8001, 8002]
    assert captured == [(200, b"complete", {"Content-Type": "application/json"})]


def test_proxy_queues_a_bounded_overload_and_retries_until_available() -> None:
    handler = object.__new__(_ProxyHandler)
    pool = _BackendPool((_Backend("127.0.0.1", 8001),), 1, None)
    handler.server = type("Server", (), {"pool": pool})()
    handler.path = "/v1/chat/completions"
    handler.command = "POST"
    handler.rfile = io.BytesIO(b"{}")
    handler.headers = Message()
    handler.headers["Content-Length"] = "2"
    responses = iter(((429, b"overloaded", {}), (200, b"complete", {})))
    captured: list[tuple[int, bytes, dict[str, str]]] = []

    handler._request_backend = MethodType(lambda self, backend, body: next(responses), handler)
    handler._write_response = MethodType(
        lambda self, status, body, headers: captured.append((status, body, headers)),
        handler,
    )

    with patch("data_designer.slurm.runtime.proxy.time.sleep") as sleep:
        handler._forward()

    sleep.assert_called_once_with(1)
    assert captured == [(200, b"complete", {})]


def test_proxy_rejects_overload_immediately_when_queue_is_disabled() -> None:
    handler = object.__new__(_ProxyHandler)
    pool = _BackendPool((_Backend("127.0.0.1", 8001),), 0, None)
    handler.server = type("Server", (), {"pool": pool})()
    handler.path = "/v1/chat/completions"
    handler.command = "POST"
    handler.rfile = io.BytesIO(b"{}")
    handler.headers = Message()
    handler.headers["Content-Length"] = "2"
    captured: list[tuple[int, bytes, dict[str, str]]] = []

    handler._request_backend = MethodType(lambda self, backend, body: (429, b"overloaded", {}), handler)
    handler._write_response = MethodType(
        lambda self, status, body, headers: captured.append((status, body, headers)),
        handler,
    )

    handler._forward()

    assert captured == [(429, b"overloaded", {})]


def test_proxy_rejects_a_truncated_fixed_length_request() -> None:
    handler = object.__new__(_ProxyHandler)
    handler.rfile = io.BytesIO(b"{")
    handler.headers = Message()
    handler.headers["Content-Length"] = "2"
    errors: list[int] = []
    handler.send_error = MethodType(lambda self, status: errors.append(status), handler)

    assert handler._read_request_body() is None
    assert errors == [400]


def test_proxy_bounds_backend_response_buffering(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status = 200

        def read(self, amount: int) -> bytes:
            assert amount == 5
            return b"12345"

        def getheaders(self) -> list[tuple[str, str]]:
            return []

    class _Connection:
        def __init__(self, host: str, port: int, timeout: int) -> None:
            del host, port, timeout

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            del method, path, body, headers

        def getresponse(self) -> _Response:
            return _Response()

        def close(self) -> None:
            pass

    handler = object.__new__(_ProxyHandler)
    handler.command = "POST"
    handler.path = "/v1/chat/completions"
    handler.headers = Message()
    monkeypatch.setattr(runtime_proxy, "_MAXIMUM_RESPONSE_BYTES", 4)
    monkeypatch.setattr(runtime_proxy.http.client, "HTTPConnection", _Connection)

    assert handler._request_backend(_Backend("127.0.0.1", 8001), b"{}") == (
        502,
        b'{"error":"backend response too large"}\n',
        {"Content-Type": "application/json"},
    )


def test_proxy_strips_connection_nominated_headers_in_both_directions(monkeypatch: pytest.MonkeyPatch) -> None:
    class _Response:
        status = 200

        def read(self, amount: int) -> bytes:
            del amount
            return b"complete"

        def getheaders(self) -> list[tuple[str, str]]:
            return [
                ("Connection", "X-Backend-Hop"),
                ("X-Backend-Hop", "private"),
                ("Content-Type", "application/json"),
            ]

    class _Connection:
        forwarded_headers: dict[str, str] = {}

        def __init__(self, host: str, port: int, timeout: int) -> None:
            del host, port, timeout

        def request(self, method: str, path: str, *, body: bytes, headers: dict[str, str]) -> None:
            del method, path, body
            type(self).forwarded_headers = headers

        def getresponse(self) -> _Response:
            return _Response()

        def close(self) -> None:
            pass

    handler = object.__new__(_ProxyHandler)
    handler.command = "POST"
    handler.path = "/v1/chat/completions"
    handler.headers = Message()
    handler.headers["Connection"] = "X-Client-Hop"
    handler.headers["X-Client-Hop"] = "private"
    handler.headers["Proxy-Connection"] = "keep-alive"
    handler.headers["Authorization"] = "Bearer reviewed"
    monkeypatch.setattr(runtime_proxy.http.client, "HTTPConnection", _Connection)

    status, body, headers = handler._request_backend(_Backend("127.0.0.1", 8001), b"{}")

    assert (status, body) == (200, b"complete")
    assert _Connection.forwarded_headers == {"Authorization": "Bearer reviewed", "Content-Length": "2"}
    assert headers == {"Content-Type": "application/json"}


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


def test_pool_selects_least_active_backend() -> None:
    pool = _BackendPool((_Backend("127.0.0.1", 8001), _Backend("127.0.0.1", 8002)), 2, 1)
    first = pool.acquire(frozenset())
    second = pool.acquire(frozenset())
    pool.release(second)
    pool.release(first)

    assert first != second
