# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import TYPE_CHECKING, Any, NoReturn

from pydantic import BaseModel

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.errors import DataDesignerError
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind

if TYPE_CHECKING:
    import litellm

logger = logging.getLogger(__name__)


def _normalize_error_detail(detail: str | None) -> str | None:
    if detail is None:
        return None
    normalized = " ".join(detail.split()).strip()
    return normalized or None


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
    return get_exception_primary_cause(exception.__cause__)


class GenerationValidationFailureError(Exception):
    summary: str
    detail: str | None
    failure_kind: str

    def __init__(
        self,
        summary: str,
        *,
        detail: str | None = None,
        failure_kind: str = "validation_error",
    ) -> None:
        self.summary = summary.strip()
        self.detail = _normalize_error_detail(detail)
        self.failure_kind = failure_kind

        message = self.summary
        if self.detail is not None:
            message = f"{message} Validation detail: {self.detail}"

        super().__init__(message)


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


class ModelGenerationValidationFailureError(DataDesignerError):
    detail: str | None
    failure_kind: str | None

    def __init__(
        self,
        message: object | None = None,
        *,
        detail: str | None = None,
        failure_kind: str | None = None,
    ) -> None:
        if message is None:
            super().__init__()
        else:
            super().__init__(message)
        self.detail = _normalize_error_detail(detail)
        self.failure_kind = failure_kind


class ImageGenerationError(DataDesignerError): ...


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
        # Canonical ProviderError from the client adapter layer
        case ProviderError():
            _raise_from_provider_error(
                exception,
                exception.kind,
                model_name,
                model_provider_name,
                purpose,
                authentication_error,
            )

        # LiteLLM-specific errors (safety net during bridge period)
        case lazy.litellm.exceptions.APIError():
            raise err_msg_parser.parse_api_error(exception, authentication_error) from None

        case lazy.litellm.exceptions.APIConnectionError():
            raise ModelAPIConnectionError(
                FormattedLLMErrorMessage(
                    cause=f"Connection to model {model_name!r} hosted on model provider {model_provider_name!r} failed while {purpose}.",
                    solution="Check your network/proxy/firewall settings.",
                )
            ) from None

        case lazy.litellm.exceptions.AuthenticationError():
            raise ModelAuthenticationError(authentication_error) from None

        case lazy.litellm.exceptions.ContextWindowExceededError():
            raise err_msg_parser.parse_context_window_exceeded_error(exception) from None

        case lazy.litellm.exceptions.UnsupportedParamsError():
            raise ModelUnsupportedParamsError(
                FormattedLLMErrorMessage(
                    cause=f"One or more of the parameters you provided were found to be unsupported by model {model_name!r} while {purpose}.",
                    solution=f"Review the documentation for model provider {model_provider_name!r} and adjust your request.",
                )
            ) from None

        case lazy.litellm.exceptions.BadRequestError():
            raise err_msg_parser.parse_bad_request_error(exception) from None

        case lazy.litellm.exceptions.InternalServerError():
            raise ModelInternalServerError(
                FormattedLLMErrorMessage(
                    cause=f"Model {model_name!r} is currently experiencing internal server issues while {purpose}.",
                    solution=f"Try again in a few moments. Check with your model provider {model_provider_name!r} if the issue persists.",
                )
            ) from None

        case lazy.litellm.exceptions.NotFoundError():
            raise ModelNotFoundError(
                FormattedLLMErrorMessage(
                    cause=f"The specified model {model_name!r} could not be found while {purpose}.",
                    solution=f"Check that the model name is correct and supported by your model provider {model_provider_name!r} and try again.",
                )
            ) from None

        case lazy.litellm.exceptions.PermissionDeniedError():
            raise ModelPermissionDeniedError(
                FormattedLLMErrorMessage(
                    cause=f"Your API key was found to lack the necessary permissions to use model {model_name!r} while {purpose}.",
                    solution=f"Use an API key that has the right permissions for the model or use a model the API key in use has access to in model provider {model_provider_name!r}.",
                )
            ) from None

        case lazy.litellm.exceptions.RateLimitError():
            raise ModelRateLimitError(
                FormattedLLMErrorMessage(
                    cause=f"You have exceeded the rate limit for model {model_name!r} while {purpose}.",
                    solution="Wait and try again in a few moments.",
                )
            ) from None

        case lazy.litellm.exceptions.Timeout():
            raise ModelTimeoutError(
                FormattedLLMErrorMessage(
                    cause=f"The request to model {model_name!r} timed out while {purpose}.",
                    solution="Check your connection and try again. You may need to increase the timeout setting for the model.",
                )
            ) from None

        case lazy.litellm.exceptions.UnprocessableEntityError():
            raise ModelUnprocessableEntityError(
                FormattedLLMErrorMessage(
                    cause=f"The request to model {model_name!r} failed despite correct request format while {purpose}.",
                    solution="This is most likely temporary. Try again in a few moments.",
                )
            ) from None

        # Parsing and validation errors
        case GenerationValidationFailureError():
            detail_text = exception.detail.rstrip(".") if exception.detail is not None else None
            validation_detail = f" Validation detail: {detail_text}." if detail_text is not None else ""
            raise ModelGenerationValidationFailureError(
                FormattedLLMErrorMessage(
                    cause=(
                        f"The model output from {model_name!r} could not be parsed into the requested format "
                        f"while {purpose}.{validation_detail}"
                    ),
                    solution="This is most likely temporary as we make additional attempts. If you continue to see more of this, simplify or modify the output schema for structured output and try again. If you are attempting token-intensive tasks like generations with high-reasoning effort, ensure that max_tokens in the model config is high enough to reach completion.",
                ),
                detail=exception.detail,
                failure_kind=exception.failure_kind,
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
    def wrapper(model_facade: Any, *args: Any, **kwargs: Any) -> Any:
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


def acatch_llm_exceptions(func: Callable) -> Callable:
    @wraps(func)
    async def wrapper(model_facade: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            return await func(model_facade, *args, **kwargs)
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
        self, exception: litellm.exceptions.APIError, auth_error_msg: FormattedLLMErrorMessage
    ) -> DataDesignerError:
        if "Error code: 403" in str(exception):
            return ModelAuthenticationError(auth_error_msg)

        return ModelAPIError(
            FormattedLLMErrorMessage(
                cause=f"An unexpected API error occurred with model {self.model_name!r} while {self.purpose}.",
                solution=f"Try again in a few moments. Check with your model provider {self.model_provider_name!r} if the issue persists.",
            )
        )


def _raise_from_provider_error(
    exception: ProviderError,
    kind: ProviderErrorKind,
    model_name: str,
    model_provider_name: str,
    purpose: str,
    authentication_error: FormattedLLMErrorMessage,
) -> NoReturn:
    """Map a canonical ProviderError to the appropriate DataDesignerError subclass."""
    _KIND_MAP: dict[ProviderErrorKind, type[DataDesignerError]] = {
        ProviderErrorKind.RATE_LIMIT: ModelRateLimitError,
        ProviderErrorKind.TIMEOUT: ModelTimeoutError,
        ProviderErrorKind.NOT_FOUND: ModelNotFoundError,
        ProviderErrorKind.PERMISSION_DENIED: ModelPermissionDeniedError,
        ProviderErrorKind.UNSUPPORTED_PARAMS: ModelUnsupportedParamsError,
        ProviderErrorKind.INTERNAL_SERVER: ModelInternalServerError,
        ProviderErrorKind.UNPROCESSABLE_ENTITY: ModelUnprocessableEntityError,
        ProviderErrorKind.API_CONNECTION: ModelAPIConnectionError,
    }

    _MESSAGES: dict[ProviderErrorKind, tuple[str, str]] = {
        ProviderErrorKind.RATE_LIMIT: (
            f"You have exceeded the rate limit for model {model_name!r} while {purpose}.",
            "Wait and try again in a few moments.",
        ),
        ProviderErrorKind.TIMEOUT: (
            f"The request to model {model_name!r} timed out while {purpose}.",
            "Check your connection and try again. You may need to increase the timeout setting for the model.",
        ),
        ProviderErrorKind.NOT_FOUND: (
            f"The specified model {model_name!r} could not be found while {purpose}.",
            f"Check that the model name is correct and supported by your model provider {model_provider_name!r} and try again.",
        ),
        ProviderErrorKind.PERMISSION_DENIED: (
            f"Your API key was found to lack the necessary permissions to use model {model_name!r} while {purpose}.",
            f"Use an API key that has the right permissions for the model or use a model the API key in use has access to in model provider {model_provider_name!r}.",
        ),
        ProviderErrorKind.UNSUPPORTED_PARAMS: (
            f"One or more of the parameters you provided were found to be unsupported by model {model_name!r} while {purpose}.",
            f"Review the documentation for model provider {model_provider_name!r} and adjust your request.",
        ),
        ProviderErrorKind.INTERNAL_SERVER: (
            f"Model {model_name!r} is currently experiencing internal server issues while {purpose}.",
            f"Try again in a few moments. Check with your model provider {model_provider_name!r} if the issue persists.",
        ),
        ProviderErrorKind.UNPROCESSABLE_ENTITY: (
            f"The request to model {model_name!r} failed despite correct request format while {purpose}.",
            "This is most likely temporary. Try again in a few moments.",
        ),
        ProviderErrorKind.API_CONNECTION: (
            f"Connection to model {model_name!r} hosted on model provider {model_provider_name!r} failed while {purpose}.",
            "Check your network/proxy/firewall settings.",
        ),
    }

    if kind == ProviderErrorKind.AUTHENTICATION:
        raise ModelAuthenticationError(authentication_error) from None

    if kind == ProviderErrorKind.CONTEXT_WINDOW_EXCEEDED:
        raise ModelContextWindowExceededError(
            FormattedLLMErrorMessage(
                cause=f"The input data for model '{model_name}' was found to exceed its supported context width while {purpose}.",
                solution="Check the model's supported max context width. Adjust the length of your input along with completions and try again.",
            )
        ) from None

    if kind == ProviderErrorKind.BAD_REQUEST:
        err_msg = FormattedLLMErrorMessage(
            cause=f"The request for model {model_name!r} was found to be malformed or missing required parameters while {purpose}.",
            solution="Check your request parameters and try again.",
        )
        if "is not a multimodal model" in str(exception):
            err_msg = FormattedLLMErrorMessage(
                cause=f"Model {model_name!r} is not a multimodal model, but it looks like you are trying to provide multimodal context while {purpose}.",
                solution="Check your request parameters and try again.",
            )
        raise ModelBadRequestError(err_msg) from None

    if kind in _KIND_MAP and kind in _MESSAGES:
        error_cls = _KIND_MAP[kind]
        cause_str, solution_str = _MESSAGES[kind]
        raise error_cls(FormattedLLMErrorMessage(cause=cause_str, solution=solution_str)) from None

    # Fallback for API_ERROR and UNSUPPORTED_CAPABILITY
    raise ModelAPIError(
        FormattedLLMErrorMessage(
            cause=f"An unexpected API error occurred with model {model_name!r} while {purpose}.",
            solution=f"Try again in a few moments. Check with your model provider {model_provider_name!r} if the issue persists.",
        )
    ) from None
