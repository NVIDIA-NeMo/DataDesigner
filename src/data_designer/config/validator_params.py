# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import Enum
from typing import Any, Optional, Union

from pydantic import Field, field_serializer, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .utils.code_lang import SQL_DIALECTS, CodeLang

SUPPORTED_CODE_LANGUAGES = {CodeLang.PYTHON, *SQL_DIALECTS}


class ValidatorType(str, Enum):
    CODE = "code"
    LOCAL_CALLABLE = "local_callable"
    REMOTE = "remote"


class CodeValidatorParams(ConfigBase):
    """Configuration for code validation. Supports Python and SQL code validation.

    Attributes:
        code_lang: The language of the code to validate. Supported values include: `python`,
            `sql:sqlite`, `sql:postgres`, `sql:mysql`, `sql:tsql`, `sql:bigquery`, `sql:ansi`.
    """

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
        validation_function: Function (`Callable[[pd.DataFrame], pd.DataFrame]`) to validate the
            data. Output must contain a column `is_valid` of type `bool`.
        output_schema: The JSON schema for the local callable validator's output. If not provided,
            the output will not be validated.
    """

    validation_function: Any = Field(
        description="Function (Callable[[pd.DataFrame], pd.DataFrame]) to validate the data"
    )
    output_schema: Optional[dict[str, Any]] = Field(
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
        endpoint_url: The URL of the remote endpoint.
        output_schema: The JSON schema for the remote validator's output. If not provided,
            the output will not be validated.
        timeout: The timeout for the HTTP request in seconds. Defaults to 30.0.
        max_retries: The maximum number of retry attempts. Defaults to 3.
        retry_backoff: The backoff factor for the retry delay in seconds. Defaults to 2.0.
        max_parallel_requests: The maximum number of parallel requests to make. Defaults to 4.
    """

    endpoint_url: str = Field(description="URL of the remote endpoint")
    output_schema: Optional[dict[str, Any]] = Field(
        default=None, description="Expected schema for remote validator's output"
    )
    timeout: float = Field(default=30.0, gt=0, description="The timeout for the HTTP request")
    max_retries: int = Field(default=3, ge=0, description="The maximum number of retry attempts")
    retry_backoff: float = Field(default=2.0, gt=1, description="The backoff factor for the retry delay")
    max_parallel_requests: int = Field(default=4, ge=1, description="The maximum number of parallel requests to make")


ValidatorParamsT: TypeAlias = Union[
    CodeValidatorParams,
    LocalCallableValidatorParams,
    RemoteValidatorParams,
]
