# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import calendar
import email.utils
import json
import time
from enum import Enum

from data_designer.engine.models.clients.types import HttpResponse


class ProviderErrorKind(str, Enum):
    API_ERROR = "api_error"
    API_CONNECTION = "api_connection"
    AUTHENTICATION = "authentication"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    QUOTA_EXCEEDED = "quota_exceeded"
    UNSUPPORTED_PARAMS = "unsupported_params"
    BAD_REQUEST = "bad_request"
    INTERNAL_SERVER = "internal_server"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


class ProviderError(Exception):
    def __init__(
        self,
        kind: ProviderErrorKind,
        message: str,
        status_code: int | None = None,
        provider_name: str | None = None,
        model_name: str | None = None,
        retry_after: float | None = None,
        cause: Exception | None = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.provider_name = provider_name
        self.model_name = model_name
        self.retry_after = retry_after
        if cause is not None:
            self.__cause__ = cause

    def __str__(self) -> str:
        return self.message

    @classmethod
    def unsupported_capability(
        cls,
        *,
        provider_name: str,
        operation: str,
        model_name: str | None = None,
        message: str | None = None,
    ) -> ProviderError:
        if message is None:
            model_segment = f" for model {model_name!r}" if model_name else ""
            message = f"Provider {provider_name!r} does not support operation {operation!r}{model_segment}."
        return cls(
            kind=ProviderErrorKind.UNSUPPORTED_CAPABILITY,
            message=message,
            provider_name=provider_name,
            model_name=model_name,
        )


def map_http_status_to_provider_error_kind(status_code: int, body_text: str = "") -> ProviderErrorKind:
    text = body_text.lower()
    if _looks_like_quota_exceeded_error(text):
        return ProviderErrorKind.QUOTA_EXCEEDED
    if status_code == 401:
        return ProviderErrorKind.AUTHENTICATION
    if status_code == 403:
        return ProviderErrorKind.PERMISSION_DENIED
    if status_code == 404:
        return ProviderErrorKind.NOT_FOUND
    if status_code == 408:
        return ProviderErrorKind.TIMEOUT
    if status_code == 413 or (status_code == 400 and _looks_like_context_window_error(text)):
        return ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED
    if status_code == 422:
        return ProviderErrorKind.UNPROCESSABLE_ENTITY
    if status_code == 429:
        return ProviderErrorKind.RATE_LIMIT
    if status_code == 400:
        if _looks_like_unsupported_params_error(text):
            return ProviderErrorKind.UNSUPPORTED_PARAMS
        return ProviderErrorKind.BAD_REQUEST
    if 500 <= status_code <= 599:
        return ProviderErrorKind.INTERNAL_SERVER
    return ProviderErrorKind.API_ERROR


def map_http_error_to_provider_error(
    *,
    response: HttpResponse,
    provider_name: str,
    model_name: str | None = None,
) -> ProviderError:
    status_code: int | None = getattr(response, "status_code", None)
    body_text = _extract_response_text(response)

    if status_code is None:
        return ProviderError(
            kind=ProviderErrorKind.API_ERROR,
            message=f"Provider {provider_name!r} request failed with an unknown HTTP status.",
            provider_name=provider_name,
            model_name=model_name,
        )

    kind = map_http_status_to_provider_error_kind(status_code=status_code, body_text=body_text)
    retry_after = _extract_retry_after(response) if status_code == 429 else None
    return ProviderError(
        kind=kind,
        message=body_text or f"Provider {provider_name!r} request failed with status code {status_code}.",
        status_code=status_code,
        provider_name=provider_name,
        model_name=model_name,
        retry_after=retry_after,
    )


def extract_message_from_exception_string(raw: str) -> str:
    """Extract a human-readable message from a stringified provider exception.

    Some providers format errors as ``"Error code: 400 - {json}"``.  This
    mirrors the structured-key lookup in ``_extract_structured_message`` but
    operates on a raw string instead of an ``HttpResponse``.
    """
    json_start = raw.find("{")
    if json_start != -1:
        try:
            payload = json.loads(raw[json_start:])
        except (json.JSONDecodeError, ValueError):
            return raw
        if isinstance(payload, dict):
            for key in ("message", "error", "detail"):
                value = payload.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
                if isinstance(value, dict):
                    nested = value.get("message")
                    if isinstance(nested, str) and nested.strip():
                        return nested.strip()
    return raw


def _extract_response_text(response: HttpResponse) -> str:
    # Try structured JSON extraction first — most providers return structured error
    # bodies and we want the human-readable message, not raw JSON.
    structured = _extract_structured_message(response)
    if structured:
        return structured

    response_text = getattr(response, "text", None)
    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    return ""


def _extract_structured_message(response: HttpResponse) -> str:
    try:
        payload = response.json()
    except Exception:
        return ""

    if isinstance(payload, dict):
        for key in ("message", "error", "detail"):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
            if isinstance(value, dict):
                nested_message = value.get("message")
                if isinstance(nested_message, str) and nested_message.strip():
                    return nested_message.strip()
            if isinstance(value, list):
                parts = [
                    item.get("msg") for item in value if isinstance(item, dict) and isinstance(item.get("msg"), str)
                ]
                if parts:
                    return "; ".join(parts)
    return ""


def _extract_retry_after(response: HttpResponse) -> float | None:
    """Parse Retry-After header value (delay-seconds or HTTP-date per RFC 7231)."""
    headers = getattr(response, "headers", None)
    if headers is None:
        return None
    raw = (
        headers.get("retry-after")
        if isinstance(headers, dict)
        else getattr(headers, "get", lambda _: None)("retry-after")
    )
    if raw is None:
        return None
    try:
        return float(raw)
    except (ValueError, TypeError):
        pass
    return _parse_http_date_as_delay(raw)


def _parse_http_date_as_delay(value: str) -> float | None:
    """Convert an HTTP-date Retry-After value to seconds from now."""
    parsed = email.utils.parsedate(value)
    if parsed is None:
        return None
    target = calendar.timegm(parsed)
    delay = target - time.time()
    return max(delay, 0.0)


def infer_error_kind_from_exception(exc: Exception) -> ProviderErrorKind:
    """Infer a ``ProviderErrorKind`` from an exception's type name.

    Used by adapters to classify transport-level exceptions (timeouts,
    connection failures, etc.) that don't carry an HTTP status code.
    """
    type_name = type(exc).__name__.lower()
    if "timeout" in type_name:
        return ProviderErrorKind.TIMEOUT
    if "connection" in type_name or "connect" in type_name:
        return ProviderErrorKind.API_CONNECTION
    if "auth" in type_name:
        return ProviderErrorKind.AUTHENTICATION
    if "ratelimit" in type_name:
        return ProviderErrorKind.RATE_LIMIT
    return ProviderErrorKind.API_ERROR


def _looks_like_context_window_error(text: str) -> bool:
    return any(
        token in text
        for token in (
            "context window",
            "context length",
            "maximum context",
            "too many tokens",
            "max tokens",
        )
    )


def _looks_like_unsupported_params_error(text: str) -> bool:
    return any(
        token in text
        for token in (
            "unsupported parameter",
            "not supported",
            "unknown parameter",
            "cannot both be specified",
            "please use only one",
            "mutually exclusive",
        )
    )


def _looks_like_quota_exceeded_error(text: str) -> bool:
    return any(
        token in text
        for token in (
            "credit balance is too low",
            "purchase credits",
            "out of credits",
            "not enough credits",
            "insufficient credits",
            "insufficient_quota",
            "insufficient quota",
            "quota exceeded",
        )
    )
