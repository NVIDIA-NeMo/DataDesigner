# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.utils.type_helpers import StrEnum
from data_designer.engine.models.clients.adapters.http_helpers import (
    parse_json_body,
    resolve_timeout,
    wrap_transport_error,
)
from data_designer.engine.models.clients.errors import map_http_error_to_provider_error
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport

if TYPE_CHECKING:
    import httpx


class ClientConcurrencyMode(StrEnum):
    SYNC = "sync"
    ASYNC = "async"


_POOL_MAX_MULTIPLIER = 2
_MIN_MAX_CONNECTIONS = 32
_MIN_KEEPALIVE_CONNECTIONS = 16


class HttpModelClient(ABC):
    """Shared HTTP transport and lifecycle logic for native model adapters.

    Each instance operates in exactly one mode — ``"sync"`` or ``"async"`` —
    set at construction time.  The mode determines which httpx client and
    transport teardown path is used.  Calling the wrong-mode methods raises
    ``RuntimeError`` immediately, preventing accidental dual-mode usage that
    leads to transport leaks and cross-mode teardown complexity.
    """

    def __init__(
        self,
        *,
        provider_name: str,
        endpoint: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        max_parallel_requests: int = 32,
        timeout_s: float = 60.0,
        concurrency_mode: ClientConcurrencyMode = ClientConcurrencyMode.SYNC,
        sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        if concurrency_mode == ClientConcurrencyMode.SYNC and async_client is not None:
            raise ValueError("async_client must not be provided for a sync-mode HttpModelClient")
        if concurrency_mode == ClientConcurrencyMode.ASYNC and sync_client is not None:
            raise ValueError("sync_client must not be provided for an async-mode HttpModelClient")

        self.provider_name = provider_name
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._retry_config = retry_config
        self._mode: ClientConcurrencyMode = concurrency_mode

        pool_max = max(_MIN_MAX_CONNECTIONS, _POOL_MAX_MULTIPLIER * max_parallel_requests)
        pool_keepalive = max(_MIN_KEEPALIVE_CONNECTIONS, max_parallel_requests)
        self._limits = lazy.httpx.Limits(
            max_connections=pool_max,
            max_keepalive_connections=pool_keepalive,
        )
        self._transport = create_retry_transport(self._retry_config)
        self._client: httpx.Client | None = sync_client
        self._aclient: httpx.AsyncClient | None = async_client
        self._init_lock = threading.Lock()
        self._closed = False

    @property
    def concurrency_mode(self) -> ClientConcurrencyMode:
        return self._mode

    @abstractmethod
    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        """Build provider-specific request headers."""

    # --- lazy client initialization ---

    def _get_sync_client(self) -> httpx.Client:
        if self._mode != ClientConcurrencyMode.SYNC:
            raise RuntimeError("Sync methods are not available on an async-mode HttpModelClient.")
        with self._init_lock:
            if self._closed:
                raise RuntimeError("Model client is closed.")
            if self._client is None:
                self._client = lazy.httpx.Client(
                    transport=self._transport,
                    limits=self._limits,
                    timeout=lazy.httpx.Timeout(self._timeout_s),
                )
            return self._client

    def _get_async_client(self) -> httpx.AsyncClient:
        if self._mode != ClientConcurrencyMode.ASYNC:
            raise RuntimeError("Async methods are not available on a sync-mode HttpModelClient.")
        with self._init_lock:
            if self._closed:
                raise RuntimeError("Model client is closed.")
            if self._aclient is None:
                self._aclient = lazy.httpx.AsyncClient(
                    transport=self._transport,
                    limits=self._limits,
                    timeout=lazy.httpx.Timeout(self._timeout_s),
                )
            return self._aclient

    # --- lifecycle ---

    def close(self) -> None:
        """Release sync-mode resources.  No-op if this is an async-mode client."""
        if self._mode != ClientConcurrencyMode.SYNC:
            return
        with self._init_lock:
            client = self._client
            transport = self._transport
            self._closed = True
            self._client = None
            self._transport = None  # type: ignore[assignment]
        if client is not None:
            client.close()
        elif transport is not None:
            transport.close()

    async def aclose(self) -> None:
        """Release async-mode resources.  No-op if this is a sync-mode client."""
        if self._mode != ClientConcurrencyMode.ASYNC:
            return
        with self._init_lock:
            async_client = self._aclient
            transport = self._transport
            self._closed = True
            self._aclient = None
            self._transport = None  # type: ignore[assignment]
        if async_client is not None:
            await async_client.aclose()
        elif transport is not None:
            await transport.aclose()

    # --- HTTP helpers ---

    def _post_sync(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        client = self._get_sync_client()
        try:
            response = client.post(
                url,
                json=payload,
                headers=headers,
                timeout=resolve_timeout(self._timeout_s, timeout),
            )
        except Exception as exc:
            raise wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return parse_json_body(response, self.provider_name, model_name)

    async def _apost(
        self,
        route: str,
        payload: dict[str, Any],
        extra_headers: dict[str, str],
        model_name: str,
        timeout: float | None = None,
    ) -> dict[str, Any]:
        url = f"{self._endpoint}{route}"
        headers = self._build_headers(extra_headers)
        async_client = self._get_async_client()
        try:
            response = await async_client.post(
                url,
                json=payload,
                headers=headers,
                timeout=resolve_timeout(self._timeout_s, timeout),
            )
        except Exception as exc:
            raise wrap_transport_error(exc, self.provider_name, model_name) from exc
        if response.status_code >= 400:
            raise map_http_error_to_provider_error(
                response=response, provider_name=self.provider_name, model_name=model_name
            )
        return parse_json_body(response, self.provider_name, model_name)
