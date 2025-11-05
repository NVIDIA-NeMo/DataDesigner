# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Callable, Literal, Optional, Type, Union

import pandas as pd
from pydantic import BaseModel, Field, field_serializer, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .errors import InvalidColumnTypeError, InvalidConfigError
from .models import ImageContext
from .sampler_params import SamplerParamsT, SamplerType
from .utils.code_lang import CodeLang
from .utils.constants import REASONING_TRACE_COLUMN_POSTFIX
from .utils.misc import assert_valid_jinja2_template, get_prompt_template_keywords
from .utils.type_helpers import SAMPLER_PARAMS, resolve_string_enum
from .validator_params import ValidatorParamsT, ValidatorType


class DataDesignerColumnType(str, Enum):
    SAMPLER = "sampler"
    LLM_TEXT = "llm-text"
    LLM_CODE = "llm-code"
    LLM_STRUCTURED = "llm-structured"
    LLM_JUDGE = "llm-judge"
    EXPRESSION = "expression"
    VALIDATION = "validation"
    SEED_DATASET = "seed-dataset"
    CUSTOM = "custom"

    @staticmethod
    def get_display_order() -> list[Self]:
        return [
            DataDesignerColumnType.SEED_DATASET,
            DataDesignerColumnType.SAMPLER,
            DataDesignerColumnType.LLM_TEXT,
            DataDesignerColumnType.LLM_CODE,
            DataDesignerColumnType.LLM_STRUCTURED,
            DataDesignerColumnType.LLM_JUDGE,
            DataDesignerColumnType.VALIDATION,
            DataDesignerColumnType.EXPRESSION,
            DataDesignerColumnType.CUSTOM,
        ]

    @property
    def has_prompt_templates(self) -> bool:
        return self in [self.LLM_TEXT, self.LLM_CODE, self.LLM_STRUCTURED, self.LLM_JUDGE]

    @property
    def is_dag_column_type(self) -> bool:
        return self in [
            self.EXPRESSION,
            self.LLM_CODE,
            self.LLM_JUDGE,
            self.LLM_STRUCTURED,
            self.LLM_TEXT,
            self.VALIDATION,
            self.CUSTOM,
        ]


class SingleColumnConfig(ConfigBase, ABC):
    name: str
    drop: bool = False

    @property
    @abstractmethod
    def column_type(self) -> DataDesignerColumnType: ...

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class SamplerColumnConfig(SingleColumnConfig):
    sampler_type: SamplerType
    params: SamplerParamsT
    conditional_params: dict[str, SamplerParamsT] = {}
    convert_to: Optional[str] = None

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.SAMPLER


class LLMTextColumnConfig(SingleColumnConfig):
    prompt: str
    model_alias: str
    system_prompt: Optional[str] = None
    multi_modal_context: Optional[list[ImageContext]] = None

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.LLM_TEXT

    @property
    def required_columns(self) -> list[str]:
        required_cols = list(get_prompt_template_keywords(self.prompt))
        if self.system_prompt:
            required_cols.extend(list(get_prompt_template_keywords(self.system_prompt)))
        return list(set(required_cols))

    @property
    def side_effect_columns(self) -> list[str]:
        return [f"{self.name}{REASONING_TRACE_COLUMN_POSTFIX}"]

    @model_validator(mode="after")
    def assert_prompt_valid_jinja(self) -> Self:
        assert_valid_jinja2_template(self.prompt)
        if self.system_prompt:
            assert_valid_jinja2_template(self.system_prompt)
        return self


class LLMCodeColumnConfig(LLMTextColumnConfig):
    code_lang: CodeLang

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.LLM_CODE


class LLMStructuredColumnConfig(LLMTextColumnConfig):
    output_format: Union[dict, Type[BaseModel]]

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.LLM_STRUCTURED

    @model_validator(mode="after")
    def validate_output_format(self) -> Self:
        if not isinstance(self.output_format, dict) and issubclass(self.output_format, BaseModel):
            self.output_format = self.output_format.model_json_schema()
        return self


class Score(ConfigBase):
    name: str = Field(..., description="A clear name for this score.")
    description: str = Field(..., description="An informative and detailed assessment guide for using this score.")
    options: dict[Union[int, str], str] = Field(..., description="Score options in the format of {score: description}.")


class LLMJudgeColumnConfig(LLMTextColumnConfig):
    scores: list[Score] = Field(..., min_length=1)

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.LLM_JUDGE


class ExpressionColumnConfig(SingleColumnConfig):
    name: str
    expr: str
    dtype: Literal["int", "float", "str", "bool"] = "str"

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.EXPRESSION

    @property
    def required_columns(self) -> list[str]:
        return list(get_prompt_template_keywords(self.expr))

    @model_validator(mode="after")
    def assert_expression_valid_jinja(self) -> Self:
        if not self.expr.strip():
            raise InvalidConfigError(
                f"🛑 Expression column '{self.name}' has an empty or whitespace-only expression. Please provide a valid Jinja2 expression (e.g., '{{ column_name }}' or '{{ col1 }} + {{ col2 }}') or remove this column if not needed."
            )
        assert_valid_jinja2_template(self.expr)
        return self


class ValidationColumnConfig(SingleColumnConfig):
    target_columns: list[str]
    validator_type: ValidatorType
    validator_params: ValidatorParamsT
    batch_size: int = Field(default=10, ge=1, description="Number of records to process in each batch")

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.VALIDATION

    @property
    def required_columns(self) -> list[str]:
        return self.target_columns


class SeedDatasetColumnConfig(SingleColumnConfig):
    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.SEED_DATASET


class CustomColumnConfig(SingleColumnConfig):
    generator_function: Callable[[pd.DataFrame], pd.DataFrame]

    @property
    def column_type(self) -> DataDesignerColumnType:
        return DataDesignerColumnType.CUSTOM

    @field_serializer("generator_function")
    def serialize_generator_function(self, v: Callable[[pd.DataFrame], pd.DataFrame]) -> str:
        return v.__name__


COLUMN_TYPE_EMOJI_MAP = {
    "general": "⚛️",  # possible analysis column type
    DataDesignerColumnType.EXPRESSION: "🧩",
    DataDesignerColumnType.LLM_CODE: "💻",
    DataDesignerColumnType.LLM_JUDGE: "⚖️",
    DataDesignerColumnType.LLM_STRUCTURED: "🗂️",
    DataDesignerColumnType.LLM_TEXT: "📝",
    DataDesignerColumnType.SEED_DATASET: "🌱",
    DataDesignerColumnType.SAMPLER: "🎲",
    DataDesignerColumnType.VALIDATION: "🔍",
    DataDesignerColumnType.CUSTOM: "🛠️",
}


ColumnConfigT: TypeAlias = Union[
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
    CustomColumnConfig,
]


def get_column_config_from_kwargs(name: str, column_type: DataDesignerColumnType, **kwargs) -> ColumnConfigT:
    """Create a Data Designer column config object from kwargs.

    Args:
        name: Name of the column.
        column_type: Type of the column.
        **kwargs: Keyword arguments to pass to the column constructor.

    Returns:
        Data Designer column object of the appropriate type.
    """
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    if column_type == DataDesignerColumnType.LLM_TEXT:
        return LLMTextColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.LLM_CODE:
        return LLMCodeColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.LLM_STRUCTURED:
        return LLMStructuredColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.LLM_JUDGE:
        return LLMJudgeColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.VALIDATION:
        return ValidationColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.EXPRESSION:
        return ExpressionColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.SAMPLER:
        return SamplerColumnConfig(name=name, **_resolve_sampler_kwargs(name, kwargs))
    elif column_type == DataDesignerColumnType.SEED_DATASET:
        return SeedDatasetColumnConfig(name=name, **kwargs)
    elif column_type == DataDesignerColumnType.CUSTOM:
        return CustomColumnConfig(name=name, **kwargs)
    raise InvalidColumnTypeError(f"🛑 {column_type} is not a valid column type.")  # pragma: no cover


def _resolve_sampler_kwargs(name: str, kwargs: dict) -> dict:
    if "sampler_type" not in kwargs:
        raise InvalidConfigError(f"🛑 `sampler_type` is required for sampler column '{name}'.")
    sampler_type = resolve_string_enum(kwargs["sampler_type"], SamplerType)

    # Handle params - it could be a dict or already a concrete object
    params_value = kwargs.get("params", {})
    expected_params_class = SAMPLER_PARAMS[sampler_type.value]

    if isinstance(params_value, expected_params_class):
        # params is already a concrete object of the right type
        params = params_value
    elif isinstance(params_value, dict):
        # params is a dictionary, create new instance
        params = expected_params_class(**params_value)
    else:
        # params is neither dict nor expected type
        raise InvalidConfigError(
            f"🛑 Invalid params for sampler column '{name}'. Expected a dictionary or an instance of {expected_params_class.__name__}."
        )

    return {
        "sampler_type": sampler_type,
        "params": params,
        **{k: v for k, v in kwargs.items() if k not in ["sampler_type", "params"]},
    }
