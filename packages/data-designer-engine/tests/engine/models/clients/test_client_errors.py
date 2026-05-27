# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from data_designer.engine.models.clients.errors import (
    ProviderError,
    ProviderErrorKind,
    extract_message_from_exception_string,
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
        (
            400,
            "Your credit balance is too low to access the Anthropic API. Please purchase credits.",
            ProviderErrorKind.QUOTA_EXCEEDED,
        ),
        (
            400,
            "`temperature` and `top_p` cannot both be specified for this model. Please use only one.",
            ProviderErrorKind.UNSUPPORTED_PARAMS,
        ),
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
        (
            400,
            "",
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": "`temperature` and `top_p` cannot both be specified for this model. Please use only one.",
                }
            },
            ProviderErrorKind.UNSUPPORTED_PARAMS,
            "`temperature` and `top_p` cannot both be specified for this model. Please use only one.",
        ),
        (
            400,
            "",
            {
                "error": {
                    "type": "invalid_request_error",
                    "message": "Your credit balance is too low to access the Anthropic API.",
                }
            },
            ProviderErrorKind.QUOTA_EXCEEDED,
            "Your credit balance is too low to access the Anthropic API.",
        ),
        (
            422,
            "",
            {
                "detail": [
                    {"loc": ["body", "name"], "msg": "field required"},
                    {"loc": ["body", "age"], "msg": "not a valid integer"},
                ]
            },
            ProviderErrorKind.UNPROCESSABLE_ENTITY,
            "field required; not a valid integer",
        ),
    ],
    ids=[
        "text-when-no-json",
        "json-over-raw-text",
        "json-when-text-missing",
        "nested-error-message",
        "mutually-exclusive-params",
        "quota-exceeded-message",
        "fastapi-list-detail",
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


def test_map_http_error_extracts_retry_after_from_http_date() -> None:
    response = StubHttpResponse(
        status_code=429,
        text="Rate limit hit",
        headers={"retry-after": "Fri, 31 Dec 2027 23:59:59 GMT"},
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.retry_after is not None
    assert error.retry_after > 0


def test_map_http_error_retry_after_returns_none_for_garbage() -> None:
    response = StubHttpResponse(
        status_code=429,
        text="Rate limit hit",
        headers={"retry-after": "not-a-date-or-number"},
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.retry_after is None


@pytest.mark.parametrize(
    "raw,expected",
    [
        (
            "Error code: 400 - {'error': {'message': 'Context length exceeded', 'type': 'invalid_request_error'}}".replace(
                "'", '"'
            ),
            "Context length exceeded",
        ),
        (
            'Error code: 403 - {"error": "Insufficient permissions"}',
            "Insufficient permissions",
        ),
        (
            'Error code: 500 - {"message": "Internal failure"}',
            "Internal failure",
        ),
        (
            'Error code: 422 - {"detail": "Unprocessable entity"}',
            "Unprocessable entity",
        ),
        (
            "Connection timed out",
            "Connection timed out",
        ),
        (
            "Error code: 400 - {not valid json",
            "Error code: 400 - {not valid json",
        ),
    ],
    ids=[
        "nested-error-message",
        "top-level-error-string",
        "top-level-message-string",
        "top-level-detail-string",
        "no-json-passthrough",
        "malformed-json-passthrough",
    ],
)
def test_extract_message_from_exception_string(raw: str, expected: str) -> None:
    assert extract_message_from_exception_string(raw) == expected
