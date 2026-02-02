# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel

from data_designer.engine.errors import DataDesignerError
from data_designer.lazy_heavy_imports import litellm

if TYPE_CHECKING:
    import litellm

logger = logging.getLogger(__name__)


def get_exception_primary_cause(exception: BaseException) -> BaseException:
    """Returns the primary cause of an exception by walking backwards.

    This recursive walkback halts when it arrives at an exception which
    has no provided __cause__ (e.g. __cause__ is None).

    Args:
        exception (Exception): An exception to start from.

    Raises:
        RecursionError: if for some reason exceptions have circular
            dependencies (seems impossible in practice).
    """
    if exception.__cause__ is None:
        return exception
    else:
        return get_exception_primary_cause(exception.__cause__)


class GenerationValidationFailureError(Exception): ...


class ModelRateLimitError(DataDesignerError): ...


class ModelTimeoutError(DataDesignerError): ...


class ModelContextWindowExceededError(DataDesignerError): ...


class ModelAuthenticationError(DataDesignerError): ...


class ModelPermissionDeniedError(DataDesignerError): ...


class ModelNotFoundError(DataDesignerError): ...


class ModelUnsupportedParamsError(DataDesignerError): ...


class ModelBadRequestError(DataDesignerError): ...


class ModelInternalServerError(DataDesignerError): ...


class ModelAPIError(DataDesignerError): ...


class ModelUnprocessableEntityError(DataDesignerError): ...


class ModelAPIConnectionError(DataDesignerError): ...


class ModelStructuredOutputError(DataDesignerError): ...


class ModelGenerationValidationFailureError(DataDesignerError): ...


class FormattedLLMErrorMessage(BaseModel):
    cause: str
    solution: str

    def __str__(self) -> str:
        return "\n".join(
            [
                "  |----------",
                f"  | Cause: {self.cause}",
                f"  | Solution: {self.solution}",
                "  |----------",
            ]
        )


def handle_llm_exceptions(
    exception: Exception, model_name: str, model_provider_name: str, purpose: str | None = None
) -> None:
    """Handle LLM-related exceptions and convert them to appropriate DataDesignerError errors.

    This method centralizes the exception handling logic for LLM operations,
    making it reusable across different contexts.

    Args:
        exception: The exception that was raised
        model_name: Name of the model that was being used
        model_provider_name: Name of the model provider that was being used
        purpose: The purpose of the model usage to show as context in the error message
    Raises:
        DataDesignerError: A more user-friendly error with appropriate error type and message
    """
    purpose = purpose or "running generation"
    authentication_error = FormattedLLMErrorMessage(
        cause=f"The API key provided for model {model_name!r} was found to be invalid or expired while {purpose}.",
        solution=f"Verify your API key for model provider and update it in your settings for model provider {model_provider_name!r}.",
    )
    err_msg_parser = DownstreamLLMExceptionMessageParser(model_name, model_provider_name, purpose)
    match exception:
        # Common errors that can come from LiteLLM
        case litellm.exceptions.APIError():
            raise err_msg_parser.parse_api_error(exception, authentication_error) from None

        case litellm.exceptions.APIConnectionError():
            raise ModelAPIConnectionError(
                FormattedLLMErrorMessage(
                    cause=f"Connection to model {model_name!r} hosted on model provider {model_provider_name!r} failed while {purpose}.",
                    solution="Check your network/proxy/firewall settings.",
                )
            ) from None

        case litellm.exceptions.AuthenticationError():
            raise ModelAuthenticationError(authentication_error) from None

        case litellm.exceptions.ContextWindowExceededError():
            raise err_msg_parser.parse_context_window_exceeded_error(exception) from None

        case litellm.exceptions.UnsupportedParamsError():
            raise ModelUnsupportedParamsError(
                FormattedLLMErrorMessage(
                    cause=f"One or more of the parameters you provided were found to be unsupported by model {model_name!r} while {purpose}.",
                    solution=f"Review the documentation for model provider {model_provider_name!r} and adjust your request.",
                )
            ) from None

        case litellm.exceptions.BadRequestError():
            raise err_msg_parser.parse_bad_request_error(exception) from None

        case litellm.exceptions.InternalServerError():
            raise ModelInternalServerError(
                FormattedLLMErrorMessage(
                    cause=f"Model {model_name!r} is currently experiencing internal server issues while {purpose}.",
                    solution=f"Try again in a few moments. Check with your model provider {model_provider_name!r} if the issue persists.",
                )
            ) from None

        case litellm.exceptions.NotFoundError():
            raise ModelNotFoundError(
                FormattedLLMErrorMessage(
                    cause=f"The specified model {model_name!r} could not be found while {purpose}.",
                    solution=f"Check that the model name is correct and supported by your model provider {model_provider_name!r} and try again.",
                )
            ) from None

        case litellm.exceptions.PermissionDeniedError():
            raise ModelPermissionDeniedError(
                FormattedLLMErrorMessage(
                    cause=f"Your API key was found to lack the necessary permissions to use model {model_name!r} while {purpose}.",
                    solution=f"Use an API key that has the right permissions for the model or use a model the API key in use has access to in model provider {model_provider_name!r}.",
                )
            ) from None

        case litellm.exceptions.RateLimitError():
            raise ModelRateLimitError(
                FormattedLLMErrorMessage(
                    cause=f"You have exceeded the rate limit for model {model_name!r} while {purpose}.",
                    solution="Wait and try again in a few moments.",
                )
            ) from None

        case litellm.exceptions.Timeout():
            raise ModelTimeoutError(
                FormattedLLMErrorMessage(
                    cause=f"The request to model {model_name!r} timed out while {purpose}.",
                    solution="Check your connection and try again. You may need to increase the timeout setting for the model.",
                )
            ) from None

        case litellm.exceptions.UnprocessableEntityError():
            raise ModelUnprocessableEntityError(
                FormattedLLMErrorMessage(
                    cause=f"The request to model {model_name!r} failed despite correct request format while {purpose}.",
                    solution="This is most likely temporary. Try again in a few moments.",
                )
            ) from None

        # Parsing and validation errors
        case GenerationValidationFailureError():
            raise ModelGenerationValidationFailureError(
                FormattedLLMErrorMessage(
                    cause=f"The provided output schema was unable to be parsed from model {model_name!r} responses while {purpose}.",
                    solution="This is most likely temporary as we make additional attempts. If you continue to see more of this, simplify or modify the output schema for structured output and try again. If you are attempting token-intensive tasks like generations with high-reasoning effort, ensure that max_tokens in the model config is high enough to reach completion.",
                )
            ) from None

        case DataDesignerError():
            raise exception from None

        case _:
            raise DataDesignerError(
                FormattedLLMErrorMessage(
                    cause=f"An unexpected error occurred while {purpose}.",
                    solution=f"Review the stack trace for more details: {exception}",
                )
            ) from exception


def catch_llm_exceptions(func: Callable) -> Callable:
    """This decorator should be used on any `ModelFacade` method that could potentially raise
    exceptions that should turn into upstream user-facing errors.
    """

    @wraps(func)
    def wrapper(model_facade: Any, *args, **kwargs):
        try:
            return func(model_facade, *args, **kwargs)
        except Exception as e:
            logger.debug(
                "\n".join(
                    [
                        "",
                        "|----------",
                        f"| Caught an exception downstream of type {type(e)!r}. Re-raising it below as a custom error with more context.",
                        "|----------",
                    ]
                ),
                exc_info=True,
                stack_info=True,
            )
            handle_llm_exceptions(
                e, model_facade.model_name, model_facade.model_provider_name, purpose=kwargs.get("purpose")
            )

    return wrapper


class DownstreamLLMExceptionMessageParser:
    def __init__(self, model_name: str, model_provider_name: str, purpose: str):
        self.model_name = model_name
        self.model_provider_name = model_provider_name
        self.purpose = purpose

    def parse_bad_request_error(self, exception: litellm.exceptions.BadRequestError) -> DataDesignerError:
        err_msg = FormattedLLMErrorMessage(
            cause=f"The request for model {self.model_name!r} was found to be malformed or missing required parameters while {self.purpose}.",
            solution="Check your request parameters and try again.",
        )
        if "is not a multimodal model" in str(exception):
            err_msg = FormattedLLMErrorMessage(
                cause=f"Model {self.model_name!r} is not a multimodal model, but it looks like you are trying to provide multimodal context while {self.purpose}.",
                solution="Check your request parameters and try again.",
            )
        return ModelBadRequestError(err_msg)

    def parse_context_window_exceeded_error(
        self, exception: litellm.exceptions.ContextWindowExceededError
    ) -> DataDesignerError:
        cause = f"The input data for model '{self.model_name}' was found to exceed its supported context width while {self.purpose}."
        try:
            if "OpenAIException - This model's maximum context length is " in str(exception):
                openai_exception_cause = (
                    str(exception).split("OpenAIException - ")[1].split("\n")[0].split(" Please reduce ")[0]
                )
                cause = f"{cause} {openai_exception_cause}"
        except Exception:
            pass
        finally:
            return ModelContextWindowExceededError(
                FormattedLLMErrorMessage(
                    cause=cause,
                    solution="Check the model's supported max context width. Adjust the length of your input along with completions and try again.",
                )
            )

    def parse_api_error(
        self, exception: litellm.exceptions.InternalServerError, auth_error_msg: FormattedLLMErrorMessage
    ) -> DataDesignerError:
        if "Error code: 403" in str(exception):
            return ModelAuthenticationError(auth_error_msg)

        return ModelAPIError(
            FormattedLLMErrorMessage(
                cause=f"An unexpected API error occurred with model {self.model_name!r} while {self.purpose}.",
                solution=f"Try again in a few moments. Check with your model provider {self.model_provider_name!r} if the issue persists.",
            )
        )
