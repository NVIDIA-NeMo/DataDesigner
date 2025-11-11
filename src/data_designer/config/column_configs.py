# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Literal, Optional, Type, Union

from pydantic import BaseModel, Field, model_validator
from typing_extensions import Self

from .base import ConfigBase
from .errors import InvalidConfigError
from .models import ImageContext
from .sampler_params import SamplerParamsT, SamplerType
from .utils.code_lang import CodeLang
from .utils.constants import REASONING_TRACE_COLUMN_POSTFIX
from .utils.misc import assert_valid_jinja2_template, get_prompt_template_keywords
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
