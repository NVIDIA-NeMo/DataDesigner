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
    def __init__(self, *, status_code: int, text: str = "", json_payload: dict[str, Any] | None = None) -> None:
        self.status_code = status_code
        self.text = text
        self._json_payload = json_payload

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


def test_map_http_error_to_provider_error_uses_text_payload() -> None:
    response = StubHttpResponse(status_code=429, text="Rate limit hit")
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider", model_name="stub-model")
    assert isinstance(error, ProviderError)
    assert error.kind == ProviderErrorKind.RATE_LIMIT
    assert error.message == "Rate limit hit"
    assert error.status_code == 429
    assert error.provider_name == "stub-provider"
    assert error.model_name == "stub-model"


def test_map_http_error_to_provider_error_uses_json_payload_when_text_missing() -> None:
    response = StubHttpResponse(
        status_code=403,
        text="",
        json_payload={"error": "Insufficient permissions for model"},
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.kind == ProviderErrorKind.PERMISSION_DENIED
    assert error.message == "Insufficient permissions for model"


def test_map_http_error_to_provider_error_uses_nested_error_message_payload() -> None:
    response = StubHttpResponse(
        status_code=400,
        text="",
        json_payload={
            "error": {
                "type": "invalid_request_error",
                "message": "The request payload is invalid.",
            }
        },
    )
    error = map_http_error_to_provider_error(response=response, provider_name="stub-provider")
    assert error.kind == ProviderErrorKind.BAD_REQUEST
    assert error.message == "The request payload is invalid."


def test_provider_error_unsupported_capability_helper() -> None:
    error = ProviderError.unsupported_capability(
        provider_name="stub-provider",
        operation="image-generation",
        model_name="stub-model",
    )
    assert error.kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY
    assert "image-generation" in error.message
    assert "stub-model" in error.message
