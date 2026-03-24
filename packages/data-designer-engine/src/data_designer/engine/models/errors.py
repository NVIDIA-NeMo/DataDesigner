# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from collections.abc import Callable
from functools import wraps
from typing import Any, NoReturn

from pydantic import BaseModel

from data_designer.engine.errors import DataDesignerError
from data_designer.engine.models.clients.errors import ProviderError, ProviderErrorKind

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


class ModelQuotaExceededError(DataDesignerError): ...


class ModelTimeoutError(DataDesignerError): ...


class ModelContextWindowExceededError(DataDesignerError): ...


class ModelAuthenticationError(DataDesignerError): ...


class ModelPermissionDeniedError(DataDesignerError): ...


class ModelNotFoundError(DataDesignerError): ...


class ModelUnsupportedParamsError(DataDesignerError): ...


class ModelBadRequestError(DataDesignerError): ...


class ModelInternalServerError(DataDesignerError): ...


class ModelUnsupportedCapabilityError(DataDesignerError): ...


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
    provider_message: str | None = None

    def __str__(self) -> str:
        lines = ["  |----------"]
        if self.provider_message is not None:
            lines.append(f"  | Provider message: {self.provider_message}")
        lines.append(f"  | Cause: {self.cause}")
        lines.extend(
            [
                f"  | Solution: {self.solution}",
                "  |----------",
            ]
        )
        return "\n".join(lines)


def _attach_provider_message(
    formatted_message: FormattedLLMErrorMessage,
    exception: ProviderError,
) -> FormattedLLMErrorMessage:
    if exception.status_code != 400:
        return formatted_message
    normalized = _normalize_error_detail(exception.message)
    if normalized is None:
        return formatted_message
    return formatted_message.model_copy(update={"provider_message": normalized})


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
        ProviderErrorKind.QUOTA_EXCEEDED: ModelQuotaExceededError,
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
        cause = (
            f"The input data for model '{model_name}' was found to exceed its supported context width while {purpose}."
        )
        context_detail = _extract_context_window_detail(str(exception))
        if context_detail:
            cause = f"{cause} {context_detail}"
        raise ModelContextWindowExceededError(
            FormattedLLMErrorMessage(
                cause=cause,
                solution="Check the model's supported max context width. Adjust the length of your input along with completions and try again.",
            )
        ) from None

    if kind == ProviderErrorKind.QUOTA_EXCEEDED:
        raise ModelQuotaExceededError(
            FormattedLLMErrorMessage(
                cause=(
                    f"Model provider {model_provider_name!r} reported insufficient credits or quota for model "
                    f"{model_name!r} while {purpose}."
                ),
                solution=f"Add credits or increase quota/billing for model provider {model_provider_name!r} and try again.",
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
        raise ModelBadRequestError(_attach_provider_message(err_msg, exception)) from None

    if kind in _KIND_MAP and kind in _MESSAGES:
        error_cls = _KIND_MAP[kind]
        cause_str, solution_str = _MESSAGES[kind]
        raise error_cls(
            _attach_provider_message(
                FormattedLLMErrorMessage(cause=cause_str, solution=solution_str),
                exception,
            )
        ) from None

    if kind == ProviderErrorKind.UNSUPPORTED_CAPABILITY:
        raise ModelUnsupportedCapabilityError(
            FormattedLLMErrorMessage(
                cause=f"{exception.message.rstrip('.')} while {purpose}.",
                solution=(
                    f"Use a model provider that supports this operation, or switch to a different model on "
                    f"{model_provider_name!r} that supports it."
                ),
            )
        ) from None

    # Fallback for API_ERROR and other unhandled kinds
    raise ModelAPIError(
        _attach_provider_message(
            FormattedLLMErrorMessage(
                cause=f"An unexpected API error occurred with model {model_name!r} while {purpose}.",
                solution=f"Try again in a few moments. Check with your model provider {model_provider_name!r} if the issue persists.",
            ),
            exception,
        )
    ) from None


def _extract_context_window_detail(error_text: str) -> str | None:
    """Extract the specific token-count detail from an OpenAI-style context window error."""
    try:
        marker = "This model's maximum context length is "
        if marker in error_text:
            detail = error_text.split(marker, 1)[1].split("\n")[0].split(" Please reduce ")[0]
            return f"{marker}{detail}"
    except Exception:
        pass
    return None
