# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Public-API HTTPX transport sharding."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import httpx

_DEFAULT_SHARD_COUNT = 16


def _split_limit(total: int, shard_count: int) -> list[int]:
    base, remainder = divmod(total, shard_count)
    return [base + (index < remainder) for index in range(shard_count)]


class ShardedAsyncHTTPTransport(httpx.AsyncBaseTransport):
    """Distribute requests round-robin across ordinary HTTPX transports."""

    def __init__(
        self,
        *,
        limits: httpx.Limits,
        shard_count: int = _DEFAULT_SHARD_COUNT,
        transport_factory: Callable[[httpx.Limits], httpx.AsyncBaseTransport] | None = None,
    ) -> None:
        if shard_count <= 0:
            raise ValueError("shard_count must be positive")
        if limits.max_connections is None or limits.max_keepalive_connections is None:
            raise ValueError("sharded transport requires finite connection limits")
        if limits.max_connections < shard_count or limits.max_keepalive_connections < shard_count:
            raise ValueError("connection limits must provide at least one connection per shard")

        factory = transport_factory or (lambda shard_limits: httpx.AsyncHTTPTransport(limits=shard_limits))
        maximums = _split_limit(limits.max_connections, shard_count)
        keepalives = _split_limit(limits.max_keepalive_connections, shard_count)
        self._shard_limits = tuple(
            httpx.Limits(
                max_connections=maximum,
                max_keepalive_connections=keepalive,
                keepalive_expiry=limits.keepalive_expiry,
            )
            for maximum, keepalive in zip(maximums, keepalives, strict=True)
        )
        self._transports = tuple(factory(shard_limits) for shard_limits in self._shard_limits)
        self._next_transport = 0
        self._closed = False

    @property
    def shard_limits(self) -> tuple[httpx.Limits, ...]:
        return self._shard_limits

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        if self._closed:
            raise RuntimeError("Transport is closed.")
        transport = self._transports[self._next_transport]
        self._next_transport = (self._next_transport + 1) % len(self._transports)
        return await transport.handle_async_request(request)

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        results = await asyncio.gather(*(transport.aclose() for transport in self._transports), return_exceptions=True)
        for result in results:
            if isinstance(result, BaseException):
                raise result
