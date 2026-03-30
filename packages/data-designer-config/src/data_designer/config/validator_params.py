# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_serializer, model_validator
from typing_extensions import Self, TypeAlias

from data_designer.config.base import ConfigBase
from data_designer.config.utils.code_lang import SQL_DIALECTS, CodeLang

SUPPORTED_CODE_LANGUAGES = {CodeLang.PYTHON, *SQL_DIALECTS}


class ValidatorType(str, Enum):
    CODE = "code"
    LOCAL_CALLABLE = "local_callable"
    REMOTE = "remote"


class CodeValidatorParams(ConfigBase):
    """Configuration for code validation. Supports Python and SQL code validation.

    Attributes:
        code_lang (required): The language of the code to validate. Supported values include: `python`,
            `sql:sqlite`, `sql:postgres`, `sql:mysql`, `sql:tsql`, `sql:bigquery`, `sql:ansi`.
    """

    validator_type: Literal[ValidatorType.CODE] = Field(
        default=ValidatorType.CODE,
        description="Validator type discriminator, always 'code' for this validator",
    )
    code_lang: CodeLang = Field(description="The language of the code to validate")

    @model_validator(mode="after")
    def validate_code_lang(self) -> Self:
        if self.code_lang not in SUPPORTED_CODE_LANGUAGES:
            raise ValueError(
                f"Unsupported code language, supported languages are: {[lang.value for lang in SUPPORTED_CODE_LANGUAGES]}"
            )
        return self


class LocalCallableValidatorParams(ConfigBase):
    """Configuration for local callable validation. Expects a function to be passed that validates the data.

    Attributes:
        validation_function (required): Function (`Callable[[pd.DataFrame], pd.DataFrame]`) to validate the
            data. Output must contain a column `is_valid` of type `bool`.
        output_schema: The JSON schema for the local callable validator's output. If not provided,
            the output will not be validated.
    """

    validator_type: Literal[ValidatorType.LOCAL_CALLABLE] = Field(
        default=ValidatorType.LOCAL_CALLABLE,
        description="Validator type discriminator, always 'local_callable' for this validator",
    )
    validation_function: Any = Field(
        description="Function (Callable[[pd.DataFrame], pd.DataFrame]) to validate the data"
    )
    output_schema: dict[str, Any] | None = Field(
        default=None, description="Expected schema for local callable validator's output"
    )

    @field_serializer("validation_function")
    def serialize_validation_function(self, v: Any) -> Any:
        return v.__name__

    @model_validator(mode="after")
    def validate_validation_function(self) -> Self:
        if not callable(self.validation_function):
            raise ValueError("Validation function must be a callable")
        return self


class RemoteValidatorParams(ConfigBase):
    """Configuration for remote validation. Sends data to a remote endpoint for validation.

    Attributes:
        endpoint_url (required): The URL of the remote endpoint.
        output_schema: The JSON schema for the remote validator's output. If not provided,
            the output will not be validated.
        timeout: The timeout for the HTTP request in seconds. Defaults to 30.0.
        max_retries: The maximum number of retry attempts. Defaults to 3.
        retry_backoff: The backoff factor for the retry delay in seconds. Defaults to 2.0.
        max_parallel_requests: The maximum number of parallel requests to make. Defaults to 4.
    """

    validator_type: Literal[ValidatorType.REMOTE] = Field(
        default=ValidatorType.REMOTE,
        description="Validator type discriminator, always 'remote' for this validator",
    )
    endpoint_url: str = Field(description="URL of the remote endpoint")
    output_schema: dict[str, Any] | None = Field(
        default=None, description="Expected schema for remote validator's output"
    )
    timeout: float = Field(default=30.0, gt=0, description="The timeout for the HTTP request")
    max_retries: int = Field(default=3, ge=0, description="The maximum number of retry attempts")
    retry_backoff: float = Field(default=2.0, gt=1, description="The backoff factor for the retry delay")
    max_parallel_requests: int = Field(default=4, ge=1, description="The maximum number of parallel requests to make")


ValidatorParamsT: TypeAlias = CodeValidatorParams | LocalCallableValidatorParams | RemoteValidatorParams
