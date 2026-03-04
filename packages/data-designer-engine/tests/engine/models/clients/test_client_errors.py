# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    map_http_error_to_provider_error,
    map_http_status_to_provider_error_kind,
)


class StubHttpResponse:
    def __init__(
        self,
        *,
        status_code: int,
        text: str = "",
        json_payload: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.status_code = status_code
        self.text = text
        self._json_payload = json_payload
        self.headers = headers or {}

    def json(self) -> dict[str, Any]:
        if self._json_payload is None:
            raise ValueError("No JSON payload")
        return self._json_payload


@pytest.mark.parametrize(
    "status_code,body_text,expected_kind",
    [
        (401, "", ProviderErrorKind.AUTHENTICATION),
        (403, "", ProviderErrorKind.PERMISSION_DENIED),
        (404, "", ProviderErrorKind.NOT_FOUND),
        (408, "", ProviderErrorKind.TIMEOUT),
        (413, "", ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED),
        (422, "", ProviderErrorKind.UNPROCESSABLE_ENTITY),
        (429, "", ProviderErrorKind.RATE_LIMIT),
        (400, "", ProviderErrorKind.BAD_REQUEST),
        (400, "maximum context length exceeded", ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED),
        (500, "", ProviderErrorKind.INTERNAL_SERVER),
        (503, "", ProviderErrorKind.INTERNAL_SERVER),
        (418, "", ProviderErrorKind.API_ERROR),
    ],
)
def test_map_http_status_to_provider_error_kind(
    status_code: int,
    body_text: str,
    expected_kind: ProviderErrorKind,
) -> None:
    assert map_http_status_to_provider_error_kind(status_code=status_code, body_text=body_text) == expected_kind


@pytest.mark.parametrize(
    "status_code,text,json_payload,expected_kind,expected_message",
    [
        (
            429,
            "Rate limit hit",
            None,
            ProviderErrorKind.RATE_LIMIT,
            "Rate limit hit",
        ),
        (
            400,
            '{"error": {"type": "invalid_request_error", "message": "Context too long."}}',
            {"error": {"type": "invalid_request_error", "message": "Context too long."}},
            ProviderErrorKind.BAD_REQUEST,
            "Context too long.",
        ),
        (
            403,
            "",
            {"error": "Insufficient permissions for model"},
            ProviderErrorKind.PERMISSION_DENIED,
            "Insufficient permissions for model",
        ),
        (
            400,
            "",
            {"error": {"type": "invalid_request_error", "message": "The request payload is invalid."}},
            ProviderErrorKind.BAD_REQUEST,
            "The request payload is invalid.",
        ),
    ],
    ids=[
        "text-when-no-json",
        "json-over-raw-text",
        "json-when-text-missing",
        "nested-error-message",
    ],
)
def test_map_http_error_to_provider_error(
    status_code: int,
    text: str,
    json_payload: dict[str, Any] | None,
    expected_kind: ProviderErrorKind,
    expected_message: str,
) -> None:
    response = StubHttpResponse(status_code=status_code, text=text, json_payload=json_payload)
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider", model_name="stub-model")
    assert isinstance(error, ProviderError)
    assert error.kind == expected_kind
    assert error.message == expected_message
    assert error.provider_name == "stub-provider"


def test_provider_error_unsupported_capability_helper() -> None:
    error = ProviderError.unsupported_capability(
        provider_name="stub-provider",
        operation="image-generation",
        model_name="stub-model",
    )
    assert error.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY
    assert "image-generation" in error.message
    assert "stub-model" in error.message


def test_provider_error_chains_cause_exception() -> None:
    original = RuntimeError("connection reset")
    error = ProviderError(
        kind=ProviderErrorKind.API_CONNECTION,
        message="Connection failed",
        cause=original,
    )
    assert error.__cause__ is original
    assert str(error) == "Connection failed"


def test_provider_error_without_cause_has_no_chain() -> None:
    error = ProviderError(
        kind=ProviderErrorKind.RATE_LIMIT,
        message="Too many requests",
    )
    assert error.__cause__ is None


def test_provider_error_is_catchable_as_exception() -> None:
    error = ProviderError(
        kind=ProviderErrorKind.AUTHENTICATION,
        message="Invalid API key",
    )
    with pytest.raises(ProviderError) as exc_info:
        raise error
    assert exc_info.value.kind == ProviderErrorKind.AUTHENTICATION
    assert str(exc_info.value) == "Invalid API key"


def test_map_http_error_extracts_retry_after_on_429() -> None:
    response = StubHttpResponse(
        status_code=429,
        text="Rate limit hit",
        headers={"retry-after": "2.5"},
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.retry_after == 2.5


def test_map_http_error_retry_after_is_none_for_non_429() -> None:
    response = StubHttpResponse(
        status_code=500,
        text="Internal server error",
        headers={"retry-after": "10"},
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.retry_after is None
