# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest
from pytest_httpx import HTTPXMock

from data_designer.config.validator_params import RemoteValidatorParams
from data_designer.engine.validators.remote import (
    RemoteEndpointClient,
    RemoteValidator,
)
from data_designer.lazy_heavy_imports import httpx

if TYPE_CHECKING:
    import httpx


@pytest.fixture()
def stub_data() -> list[dict]:
    return [{"text": "Sample text", "id": 1}]


def test_validate_with_remote_endpoint(httpx_mock: HTTPXMock, stub_data: list[dict]):
    # Setup mock response
    httpx_mock.add_response(
        method="POST", url="http://localhost:8080", json={"data": [{"is_valid": True, "confidence": "0.98"}]}
    )

    validator = RemoteValidator(
        RemoteValidatorParams(
            endpoint_url="http://localhost:8080",
        )
    )

    results = validator.run_validation(stub_data)

    # Verify results
    assert len(results) == 1
    assert results[0].is_valid is True
    assert results[0].confidence == "0.98"


def test_remote_endpoint_client(httpx_mock: HTTPXMock):
    # Add custom callback response that tests auth and parses content
    def custom_response_callback(request: httpx.Request):
        content = request.read().decode("utf-8")
        parsed_content = json.loads(content)

        return httpx.Response(status_code=200, json={"sample_text": parsed_content["sample_content"]["text"]})

    httpx_mock.add_callback(custom_response_callback)

    client = RemoteEndpointClient(
        config=RemoteValidatorParams(
            endpoint_url="http://localhost:8080",
        ),
    )
    response = client.post_to_remote_endpoint({"sample_content": {"text": ["Sample text"]}})
    assert response["sample_text"] == ["Sample text"]
