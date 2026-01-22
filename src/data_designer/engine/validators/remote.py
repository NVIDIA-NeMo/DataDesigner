# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from httpx_retries import Retry, RetryTransport

from data_designer.config.validator_params import RemoteValidatorParams
from data_designer.engine.errors import RemoteValidationSchemaError
from data_designer.engine.processing.gsonschema.exceptions import JSONSchemaValidationError
from data_designer.engine.processing.gsonschema.validators import validate
from data_designer.engine.validators.base import BaseValidator, ValidationResult
from data_designer.lazy_heavy_imports import httpx

if TYPE_CHECKING:
    import httpx

logger = logging.getLogger(__name__)


class RemoteEndpointClient:
    """Client for making parallel HTTP requests to remote endpoints with retry, timeout, and auth support."""

    def __init__(
        self,
        config: RemoteValidatorParams,
    ):
        """
        Initialize the remote endpoint client.

        Args:
            config: Remote validator parameters
        """
        self.endpoint_url = config.endpoint_url
        self.output_schema = config.output_schema
        self.timeout = config.timeout
        self.max_retries = config.max_retries
        self.retry_backoff = config.retry_backoff

    def post_to_remote_endpoint(self, content: dict) -> dict:
        """
        Make a single HTTP request with retry logic.

        Args:
            content: The content to be posted to the remote endpoint

        Returns:
            The JSON response from the remote endpoint

        Raises:
            httpx.RequestError: If all retry attempts fail
            httpx.HTTPStatusError: If the server returns an error status
        """
        retry = Retry(
            total=self.max_retries,
            backoff_factor=self.retry_backoff,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        transport = RetryTransport(retry=retry)

        with httpx.Client(
            timeout=httpx.Timeout(self.timeout),
            transport=transport,
        ) as http_client:
            response = http_client.post(
                self.endpoint_url,
                json=content,
            )
            response.raise_for_status()

            response_json = response.json()
            if self.output_schema:
                try:
                    validate(response_json, self.output_schema, no_extra_properties=True)
                except JSONSchemaValidationError as exc:
                    raise RemoteValidationSchemaError(str(exc)) from exc
            return response_json


class RemoteValidator(BaseValidator):
    def __init__(self, config: RemoteValidatorParams):
        self.remote_endpoint_client = RemoteEndpointClient(config=config)

    def run_validation(self, data: list[dict]) -> ValidationResult:
        result = self.remote_endpoint_client.post_to_remote_endpoint(content={"data": data})
        return ValidationResult.model_validate(result)
