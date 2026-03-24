# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import re
from typing import Any
from unittest.mock import MagicMock

import pytest

from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind
from data_designer.engine.models.errors import (
    DataDesignerError,
    GenerationValidationFailureError,
    ModelAPIConnectionError,
    ModelAPIError,
    ModelAuthenticationError,
    ModelBadRequestError,
    ModelContextWindowExceededError,
    ModelGenerationValidationFailureError,
    ModelInternalServerError,
    ModelNotFoundError,
    ModelPermissionDeniedError,
    ModelQuotaExceededError,
    ModelRateLimitError,
    ModelTimeoutError,
    ModelUnprocessableEntityError,
    ModelUnsupportedCapabilityError,
    ModelUnsupportedParamsError,
    catch_llm_exceptions,
    get_exception_primary_cause,
    handle_llm_exceptions,
)

stub_model_name = "test-model"
stub_model_provider_name = "nvbuild"
stub_purpose = "running generation for column 'test'"


@pytest.mark.parametrize(
    "exception,expected_exception,expected_error_msg",
    [
        (
            ProviderError(
                kind=ProviderErrorKind.BAD_REQUEST,
                message="Unexpected field 'foo' in request payload.",
                status_code=400,
            ),
            ModelBadRequestError,
            (
                f"Provider message: Unexpected field 'foo' in request payload.\n  | Cause: The request for model "
                f"'{stub_model_name}' was found to be malformed or missing required parameters while {stub_purpose}."
            ),
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.UNSUPPORTED_PARAMS,
                message="`temperature` and `top_p` cannot both be specified for this model. Please use only one.",
                status_code=400,
            ),
            ModelUnsupportedParamsError,
            (
                "Provider message: `temperature` and `top_p` cannot both be specified for this model. Please use "
                f"only one.\n  | Cause: One or more of the parameters you provided were found to be unsupported by "
                f"model '{stub_model_name}' while {stub_purpose}."
            ),
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.QUOTA_EXCEEDED,
                message="Your credit balance is too low to access the Anthropic API.",
                status_code=400,
            ),
            ModelQuotaExceededError,
            (
                f"Cause: Model provider '{stub_model_provider_name}' reported insufficient credits or quota for model "
                f"'{stub_model_name}' while {stub_purpose}."
            ),
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.UNSUPPORTED_CAPABILITY,
                message="Provider 'anthropic-prod' does not support operation 'embeddings'.",
            ),
            ModelUnsupportedCapabilityError,
            f"Cause: Provider 'anthropic-prod' does not support operation 'embeddings' while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.PERMISSION_DENIED,
                message="Missing required scope.",
                status_code=403,
            ),
            ModelPermissionDeniedError,
            f"Cause: Your API key was found to lack the necessary permissions to use model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.RATE_LIMIT,
                message="Rate limit exceeded",
                status_code=429,
            ),
            ModelRateLimitError,
            f"Cause: You have exceeded the rate limit for model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.AUTHENTICATION,
                message="Invalid API key",
                status_code=401,
            ),
            ModelAuthenticationError,
            f"Cause: The API key provided for model '{stub_model_name}' was found to be invalid or expired while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.API_CONNECTION,
                message="Connection refused",
            ),
            ModelAPIConnectionError,
            f"Cause: Connection to model '{stub_model_name}' hosted on model provider '{stub_model_provider_name}' failed while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.TIMEOUT,
                message="Request timed out",
                status_code=408,
            ),
            ModelTimeoutError,
            f"Cause: The request to model '{stub_model_name}' timed out while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.NOT_FOUND,
                message="Model not found",
                status_code=404,
            ),
            ModelNotFoundError,
            f"Cause: The specified model '{stub_model_name}' could not be found while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.INTERNAL_SERVER,
                message="Internal server error",
                status_code=500,
            ),
            ModelInternalServerError,
            f"Cause: Model '{stub_model_name}' is currently experiencing internal server issues while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.UNPROCESSABLE_ENTITY,
                message="Unprocessable entity",
                status_code=422,
            ),
            ModelUnprocessableEntityError,
            f"Cause: The request to model '{stub_model_name}' failed despite correct request format while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.API_ERROR,
                message="Unknown API error",
                status_code=418,
            ),
            ModelAPIError,
            f"Cause: An unexpected API error occurred with model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            ProviderError(
                kind=ProviderErrorKind.BAD_REQUEST,
                message=f"{stub_model_name} is not a multimodal model",
                status_code=400,
            ),
            ModelBadRequestError,
            f"Cause: Model '{stub_model_name}' is not a multimodal model, but it looks like you are trying to provide multimodal context while {stub_purpose}.",
        ),
        (
            GenerationValidationFailureError(
                "Generation validation failure",
                detail="Response doesn't match requested <response_schema>: 'name' is a required property",
            ),
            ModelGenerationValidationFailureError,
            (
                f"Cause: The model output from '{stub_model_name}' could not be parsed into the requested format "
                f"while {stub_purpose}. Validation detail: Response doesn't match requested <response_schema>: "
                "'name' is a required property."
            ),
        ),
        (
            Exception("Some unexpected error"),
            DataDesignerError,
            f"Cause: An unexpected error occurred while {stub_purpose}.",
        ),
        (DataDesignerError("Some NemoDataDesigner error"), DataDesignerError, "Some NemoDataDesigner error"),
    ],
    ids=[
        "bad_request",
        "unsupported_params",
        "quota_exceeded",
        "unsupported_capability",
        "permission_denied",
        "rate_limit",
        "authentication",
        "api_connection",
        "timeout",
        "not_found",
        "internal_server",
        "unprocessable_entity",
        "api_error",
        "bad_request_multimodal",
        "generation_validation_failure",
        "unexpected_exception",
        "data_designer_error_passthrough",
    ],
)
def test_handle_llm_exceptions(
    exception: Exception, expected_exception: type[Exception], expected_error_msg: str
) -> None:
    with pytest.raises(expected_exception, match=re.escape(expected_error_msg)):
        handle_llm_exceptions(exception, stub_model_name, stub_model_provider_name, stub_purpose)


def test_handle_llm_exceptions_preserves_generation_failure_kind() -> None:
    with pytest.raises(ModelGenerationValidationFailureError) as exc_info:
        handle_llm_exceptions(
            GenerationValidationFailureError(
                "Generation validation failure",
                detail="Response doesn't match requested <response_schema>: 'name' is a required property",
                failure_kind="schema_validation",
            ),
            stub_model_name,
            stub_model_provider_name,
            stub_purpose,
        )

    assert exc_info.value.failure_kind == "schema_validation"
    assert exc_info.value.detail == "Response doesn't match requested <response_schema>: 'name' is a required property"


def test_handle_llm_exceptions_strips_duplicate_period_from_validation_detail() -> None:
    with pytest.raises(ModelGenerationValidationFailureError, match=r"Validation detail: Field required\.") as exc_info:
        handle_llm_exceptions(
            GenerationValidationFailureError(
                "Generation validation failure",
                detail="Field required.",
                failure_kind="schema_validation",
            ),
            stub_model_name,
            stub_model_provider_name,
            stub_purpose,
        )

    assert "Field required.." not in str(exc_info.value)
    assert exc_info.value.detail == "Field required."


def test_catch_llm_exceptions() -> None:
    @catch_llm_exceptions
    def stub_function(model_facade: Any, *args: Any, **kwargs: Any) -> None:
        raise ProviderError(kind=ProviderErrorKind.RATE_LIMIT, message="Rate limit exceeded", status_code=429)

    with pytest.raises(ModelRateLimitError, match="Cause: You have exceeded the rate limit for model"):
        stub_function(MagicMock(model_name=stub_model_name, model_provider_name=stub_model_provider_name))


def test_get_exception_primary_cause_with_cause() -> None:
    root_cause = ValueError("Root cause")
    try:
        raise root_cause
    except ValueError as e:
        try:
            raise RuntimeError("Intermediate") from e
        except RuntimeError as e2:
            try:
                raise Exception("Top level") from e2
            except Exception as top_exception:
                result = get_exception_primary_cause(top_exception)
                assert result == root_cause


def test_get_exception_primary_cause_without_cause() -> None:
    exception = ValueError("No cause")
    result = get_exception_primary_cause(exception)
    assert result == exception


def test_handle_llm_exceptions_context_window_with_openai_detail() -> None:
    exception = ProviderError(
        kind=ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED,
        message="This model's maximum context length is 32768 tokens. However, you requested 32778 tokens (10 in the messages, 32768 in the completion). Please reduce the length of the messages or completion",
        status_code=400,
    )
    with pytest.raises(ModelContextWindowExceededError) as exc_info:
        handle_llm_exceptions(
            exception, model_name=stub_model_name, model_provider_name=stub_model_provider_name, purpose=stub_purpose
        )
    assert "exceed its supported context width" in str(exc_info.value)
    assert "maximum context length is 32768 tokens" in str(exc_info.value)


def test_handle_llm_exceptions_context_window_without_openai_detail() -> None:
    exception = ProviderError(
        kind=ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED,
        message="context length exceeded",
        status_code=400,
    )
    with pytest.raises(ModelContextWindowExceededError) as exc_info:
        handle_llm_exceptions(
            exception, model_name=stub_model_name, model_provider_name=stub_model_provider_name, purpose=stub_purpose
        )
    assert "exceed its supported context width" in str(exc_info.value)
    assert "maximum context length" not in str(exc_info.value)
