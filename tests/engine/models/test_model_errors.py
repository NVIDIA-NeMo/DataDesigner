# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from typing import Any
from unittest.mock import MagicMock

import pytest
from litellm.exceptions import (
    APIConnectionError,
    APIError,
    AuthenticationError,
    BadRequestError,
    ContextWindowExceededError,
    InternalServerError,
    NotFoundError,
    PermissionDeniedError,
    RateLimitError,
    Timeout,
    UnprocessableEntityError,
    UnsupportedParamsError,
)

from data_designer.engine.models.errors import (
    DataDesignerError,
    DownstreamLLMExceptionMessageParser,
    FormattedLLMErrorMessage,
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
    ModelRateLimitError,
    ModelTimeoutError,
    ModelUnprocessableEntityError,
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
            APIConnectionError("Connection error", "openai", stub_model_name),
            ModelAPIConnectionError,
            f"Cause: Connection to model '{stub_model_name}' hosted on model provider '{stub_model_provider_name}' failed while {stub_purpose}.",
        ),
        (
            APIError(500, "Some litellm error", "openai", stub_model_name),
            ModelAPIError,
            f"Cause: An unexpected API error occurred with model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            AuthenticationError("Authentication error", "openai", stub_model_name),
            ModelAuthenticationError,
            f"Cause: The API key provided for model '{stub_model_name}' was found to be invalid or expired while {stub_purpose}.",
        ),
        (
            BadRequestError("Bad request", "openai", stub_model_name),
            ModelBadRequestError,
            f"Cause: The request for model '{stub_model_name}' was found to be malformed or missing required parameters while {stub_purpose}.",
        ),
        (
            ContextWindowExceededError("Context window exceeded", "openai", stub_model_name),
            ModelContextWindowExceededError,
            f"Cause: The input data for model '{stub_model_name}' was found to exceed its supported context width while {stub_purpose}.",
        ),
        (
            InternalServerError("Internal server error", "openai", stub_model_name),
            ModelInternalServerError,
            f"Cause: Model '{stub_model_name}' is currently experiencing internal server issues while {stub_purpose}.",
        ),
        (
            NotFoundError("Not found", "openai", stub_model_name),
            ModelNotFoundError,
            f"Cause: The specified model '{stub_model_name}' could not be found while {stub_purpose}.",
        ),
        (
            PermissionDeniedError(
                "Permission denied", "openai", stub_model_name, MagicMock(status_code=403, text="Permission denied")
            ),
            ModelPermissionDeniedError,
            f"Cause: Your API key was found to lack the necessary permissions to use model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            RateLimitError("Rate limit exceeded", "openai", stub_model_name),
            ModelRateLimitError,
            f"Cause: You have exceeded the rate limit for model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            Timeout("Request timed out", "openai", stub_model_name),
            ModelTimeoutError,
            f"Cause: The request to model '{stub_model_name}' timed out while {stub_purpose}.",
        ),
        (
            UnprocessableEntityError("Unprocessable entity", "openai", stub_model_name, response=MagicMock()),
            ModelUnprocessableEntityError,
            f"Cause: The request to model '{stub_model_name}' failed despite correct request format while {stub_purpose}.",
        ),
        (
            UnsupportedParamsError("Unsupported parameters", "openai", stub_model_name),
            ModelUnsupportedParamsError,
            f"Cause: One or more of the parameters you provided were found to be unsupported by model '{stub_model_name}' while {stub_purpose}.",
        ),
        (
            GenerationValidationFailureError("Generation validation failure"),
            ModelGenerationValidationFailureError,
            f"Cause: The provided output schema was unable to be parsed from model '{stub_model_name}' responses while {stub_purpose}.",
        ),
        (
            Exception("Some unexpected error"),
            DataDesignerError,
            f"Cause: An unexpected error occurred while {stub_purpose}.",
        ),
        (DataDesignerError("Some NemoDataDesigner error"), DataDesignerError, "Some NemoDataDesigner error"),
    ],
)
def test_handle_llm_exceptions(exception, expected_exception, expected_error_msg):
    with pytest.raises(expected_exception, match=expected_error_msg):
        handle_llm_exceptions(exception, stub_model_name, stub_model_provider_name, stub_purpose)


def test_catch_llm_exceptions():
    @catch_llm_exceptions
    def stub_function(model_facade: Any, *args, **kwargs):
        raise RateLimitError("Rate limit exceeded", "openai", stub_model_name)

    with pytest.raises(ModelRateLimitError, match="Cause: You have exceeded the rate limit for model"):
        stub_function(MagicMock(model_name=stub_model_name))


def test_openai_exception_message_parser():
    parser = DownstreamLLMExceptionMessageParser(stub_model_name, stub_model_provider_name, stub_purpose)

    with pytest.raises(
        ModelBadRequestError,
        match="Cause: The request for model 'test-model' was found to be malformed or missing required parameters",
    ):
        raise parser.parse_bad_request_error(BadRequestError("Bad request", "openai", stub_model_name))

    with pytest.raises(
        ModelBadRequestError,
        match="Cause: Model 'test-model' is not a multimodal model, but it looks like you are trying to provide multimodal context",
    ):
        raise parser.parse_bad_request_error(
            BadRequestError(f"Bad request. {stub_model_name} is not a multimodal model", "openai", stub_model_name)
        )

    with pytest.raises(
        ModelContextWindowExceededError,
        match="Cause: The input data for model 'test-model' was found to exceed its supported context width",
    ):
        raise parser.parse_context_window_exceeded_error(
            ContextWindowExceededError("Context window exceeded", "openai", stub_model_name)
        )

    with pytest.raises(ModelContextWindowExceededError, match="This model's maximum context length is 32768 tokens."):
        detailed_error_from_upstream = "OpenAIException - This model's maximum context length is 32768 tokens. However, you requested 32778 tokens (10 in the messages, 32768 in the completion). Please reduce the length of the messages or completion"
        raise parser.parse_context_window_exceeded_error(
            ContextWindowExceededError(detailed_error_from_upstream, "openai", stub_model_name)
        )

    authentication_error = FormattedLLMErrorMessage(cause="Test auth error cause", solution="Test auth errorsolution")
    with pytest.raises(
        ModelAPIError,
        match="Cause: An unexpected API error occurred with model 'test-model' while running generation for column 'test'.",
    ):
        raise parser.parse_api_error(APIError(500, "Some api error", "openai", stub_model_name), authentication_error)

    with pytest.raises(ModelAuthenticationError, match="Cause: Test auth error cause"):
        raise parser.parse_api_error(
            APIError(403, "Some obtuse error. Error code: 403", "openai", stub_model_name), authentication_error
        )


def test_get_exception_primary_cause_with_cause():
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


def test_get_exception_primary_cause_without_cause():
    exception = ValueError("No cause")
    result = get_exception_primary_cause(exception)
    assert result == exception


def test_handle_llm_exceptions_context_window_with_openai_exception():
    exception = ContextWindowExceededError(
        "OpenAIException - The model's context window was exceeded. Please reduce the length of your prompt or try a different model. Please reduce the prompt length.",
        "openai",
        stub_model_name,
    )
    with pytest.raises(ModelContextWindowExceededError) as exc_info:
        handle_llm_exceptions(
            exception, model_name=stub_model_name, model_provider_name=stub_model_provider_name, purpose=stub_purpose
        )
    assert "exceed its supported context width" in str(exc_info.value)


def test_handle_llm_exceptions_context_window_with_openai_exception_parsing_error():
    class MockException(ContextWindowExceededError):
        def __str__(self):
            raise Exception("Parsing error")

    exception = MockException("Context window exceeded", "openai", stub_model_name)
    with pytest.raises(ModelContextWindowExceededError) as exc_info:
        handle_llm_exceptions(
            exception, model_name=stub_model_name, model_provider_name=stub_model_provider_name, purpose=stub_purpose
        )
    assert "exceed its supported context width" in str(exc_info.value)
