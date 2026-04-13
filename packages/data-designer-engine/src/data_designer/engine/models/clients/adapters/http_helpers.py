# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    infer_error_kind_from_exception,
)

if TYPE_CHECKING:
    import httpx


def parse_json_body(response: httpx.Response, provider_name: str, model_name: str | None) -> dict[str, Any]:
    """Parse JSON from a successful HTTP response."""
    try:
        return response.json()
    except Exception as exc:
        raise ProviderError(
            kind=ProviderErrorKind.API_ERROR,
            message=f"Provider {provider_name!r} returned a non-JSON response (status {response.status_code}).",
            status_code=response.status_code,
            provider_name=provider_name,
            model_name=model_name,
            cause=exc,
        ) from exc


def wrap_transport_error(exc: Exception, provider_name: str, model_name: str | None) -> ProviderError:
    """Convert transport exceptions into canonical ``ProviderError`` values."""
    return ProviderError(
        kind=infer_error_kind_from_exception(exc),
        message=str(exc) or f"Transport error from provider {provider_name!r}",
        provider_name=provider_name,
        model_name=model_name,
        cause=exc,
    )


def resolve_timeout(default_timeout_s: float, per_request: float | None) -> httpx.Timeout:
    """Build an ``httpx.Timeout`` from adapter defaults and request overrides."""
    return lazy.httpx.Timeout(per_request if per_request is not None else default_timeout_s)
