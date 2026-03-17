# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.models.clients.adapters.http_helpers import (
    parse_json_body,
    resolve_timeout,
    wrap_transport_error,
)
from data_designer.engine.models.clients.errors import map_http_error_to_provider_error
from data_designer.engine.models.clients.retry import RetryConfig, create_retry_transport

if TYPE_CHECKING:
    import httpx

_POOL_MAX_MULTIPLIER = 2
_MIN_MAX_CONNECTIONS = 32
_MIN_KEEPALIVE_CONNECTIONS = 16


class HttpModelClient(ABC):
    """Shared HTTP transport and lifecycle logic for native model adapters."""

    def __init__(
        self,
        *,
        provider_name: str,
        model_id: str,
        endpoint: str,
        api_key: str | None = None,
        retry_config: RetryConfig | None = None,
        max_parallel_requests: int = 32,
        timeout_s: float = 60.0,
        sync_client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.provider_name = provider_name
        self._model_id = model_id
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout_s = timeout_s
        self._retry_config = retry_config

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

    @abstractmethod
    def _build_headers(self, extra_headers: dict[str, str]) -> dict[str, str]:
        """Build provider-specific request headers."""

    def _get_sync_client(self) -> httpx.Client:
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

    def close(self) -> None:
        with self._init_lock:
            client = self._client
            self._closed = True
            self._client = None
        if client is not None:
            client.close()

    async def aclose(self) -> None:
        with self._init_lock:
            async_client = self._aclient
            sync_client = self._client
            self._closed = True
            self._aclient = None
            self._client = None
        if async_client is not None:
            await async_client.aclose()
        if sync_client is not None:
            sync_client.close()

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
