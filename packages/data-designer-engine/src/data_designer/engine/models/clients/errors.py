# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from data_designer.engine.models.clients.types import HttpResponse


class ProviderErrorKind(str, Enum):
    API_ERROR = "api_error"
    API_CONNECTION = "api_connection"
    AUTHENTICATION = "authentication"
    CONTEXT_WINDOW_EXCEEDED = "context_window_exceeded"
    UNSUPPORTED_PARAMS = "unsupported_params"
    BAD_REQUEST = "bad_request"
    INTERNAL_SERVER = "internal_server"
    NOT_FOUND = "not_found"
    PERMISSION_DENIED = "permission_denied"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    UNPROCESSABLE_ENTITY = "unprocessable_entity"
    UNSUPPORTED_CAPABILITY = "unsupported_capability"


@dataclass
class ProviderError(Exception):
    kind: ProviderErrorKind
    message: str
    status_code: int | None = None
    provider_name: str | None = None
    model_name: str | None = None
    cause: Exception | None = None

    def __post_init__(self) -> None:
        Exception.__init__(self, self.message)
        if self.cause is not None:
            self.__cause__ = self.cause

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
    return ProviderError(
        kind=kind,
        message=body_text or f"Provider {provider_name!r} request failed with status code {status_code}.",
        status_code=status_code,
        provider_name=provider_name,
        model_name=model_name,
    )


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
    return ""


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
