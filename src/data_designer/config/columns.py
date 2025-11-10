# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Literal, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .errors import InvalidColumnTypeError, InvalidConfigError
from .models import ImageContext
from .sampler_params import SamplerParamsT, SamplerType
from .utils.code_lang import CodeLang
from .utils.constants import REASONING_TRACE_COLUMN_POSTFIX
from .utils.misc import assert_valid_jinja2_template, get_prompt_template_keywords
from .utils.type_helpers import SAMPLER_PARAMS, create_str_enum_from_discriminated_type_union, resolve_string_enum
from .validator_params import ValidatorParamsT, ValidatorType


class SingleColumnConfig(ConfigBase, ABC):
    name: str
    drop: bool = False
    column_type: str

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
    column_type: Literal["sampler"] = "sampler"


class LLMTextColumnConfig(SingleColumnConfig):
    prompt: str
    model_alias: str
    system_prompt: Optional[str] = None
    multi_modal_context: Optional[list[ImageContext]] = None
    column_type: Literal["llm-text"] = "llm-text"

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
    column_type: Literal["llm-code"] = "llm-code"


class LLMStructuredColumnConfig(LLMTextColumnConfig):
    output_format: Union[dict, Type[BaseModel]]
    column_type: Literal["llm-structured"] = "llm-structured"

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
    column_type: Literal["llm-judge"] = "llm-judge"


class ExpressionColumnConfig(SingleColumnConfig):
    name: str
    expr: str
    dtype: Literal["int", "float", "str", "bool"] = "str"
    column_type: Literal["expression"] = "expression"

    @property
    def required_columns(self) -> list[str]:
        return list(get_prompt_template_keywords(self.expr))

    @model_validator(mode="after")
    def assert_expression_valid_jinja(self) -> Self:
        if not self.expr.strip():
            raise InvalidConfigError(
                f"🛑 Expression column '{self.name}' has an empty or whitespace-only expression. "
                f"Please provide a valid Jinja2 expression (e.g., '{{ column_name }}' or '{{ col1 }} + {{ col2 }}') "
                "or remove this column if not needed."
            )
        assert_valid_jinja2_template(self.expr)
        return self


class ValidationColumnConfig(SingleColumnConfig):
    target_columns: list[str]
    validator_type: ValidatorType
    validator_params: ValidatorParamsT
    batch_size: int = Field(default=10, ge=1, description="Number of records to process in each batch")
    column_type: Literal["validation"] = "validation"

    @property
    def required_columns(self) -> list[str]:
        return self.target_columns


class SeedDatasetColumnConfig(SingleColumnConfig):
    column_type: Literal["seed-dataset"] = "seed-dataset"


ColumnConfigT: TypeAlias = Union[
    ExpressionColumnConfig,
    LLMCodeColumnConfig,
    LLMJudgeColumnConfig,
    LLMStructuredColumnConfig,
    LLMTextColumnConfig,
    SamplerColumnConfig,
    SeedDatasetColumnConfig,
    ValidationColumnConfig,
]


DataDesignerColumnType = create_str_enum_from_discriminated_type_union(
    enum_name="DataDesignerColumnType",
    type_union=ColumnConfigT,
    discriminator_field_name="column_type",
)


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
}


def column_type_used_in_execution_dag(column_type: Union[str, DataDesignerColumnType]) -> bool:
    """Return True if the column type is used in the workflow execution DAG."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    return column_type in {
        DataDesignerColumnType.EXPRESSION,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.VALIDATION,
    }


def column_type_is_llm_generated(column_type: Union[str, DataDesignerColumnType]) -> bool:
    """Return True if the column type is an LLM-generated column."""
    column_type = resolve_string_enum(column_type, DataDesignerColumnType)
    return column_type in {
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
    }


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
    raise InvalidColumnTypeError(f"🛑 {column_type} is not a valid column type.")  # pragma: no cover


def get_column_display_order() -> list[DataDesignerColumnType]:
    """Return the preferred display order of the column types."""
    return [
        DataDesignerColumnType.SEED_DATASET,
        DataDesignerColumnType.SAMPLER,
        DataDesignerColumnType.LLM_TEXT,
        DataDesignerColumnType.LLM_CODE,
        DataDesignerColumnType.LLM_STRUCTURED,
        DataDesignerColumnType.LLM_JUDGE,
        DataDesignerColumnType.VALIDATION,
        DataDesignerColumnType.EXPRESSION,
    ]


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
