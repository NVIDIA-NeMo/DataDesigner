# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from typing import Annotated, Literal, Optional, Type, Union

from pydantic import BaseModel, Discriminator, Field, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ImageContext
from data_designer.config.sampler_params import SamplerParamsT, SamplerType
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.constants import REASONING_TRACE_COLUMN_POSTFIX
from data_designer.config.utils.misc import assert_valid_jinja2_template, get_prompt_template_keywords
from data_designer.config.validator_params import ValidatorParamsT, ValidatorType


class SingleColumnConfig(ConfigBase, ABC):
    """Abstract base class for all single-column configuration types.

    This class serves as the foundation for all column configurations in DataDesigner,
    defining shared fields and properties across all column types.

    Attributes:
        name: Unique name of the column to be generated.
        drop: If True, the column will be generated but removed from the final dataset.
            Useful for intermediate columns that are dependencies for other columns.
        column_type: Discriminator field that identifies the specific column type.
            Subclasses must override this field to specify the column type with a `Literal` value.
    """

    name: str
    drop: bool = False
    column_type: str

    @property
    def required_columns(self) -> list[str]:
        """Returns a list of column names that must exist before this column can be generated.

        Returns:
            List of column names that this column depends on. Empty list indicates
            no dependencies. Override in subclasses to specify dependencies.
        """
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        """Returns a list of additional columns that this column will create as a side effect.

        Some column types generate additional metadata or auxiliary columns alongside
        the primary column (e.g., reasoning traces for LLM columns).

        Returns:
            List of column names that this column will create as a side effect. Empty list
            indicates no side effect columns. Override in subclasses to specify side effects.
        """
        return []


class SamplerColumnConfig(SingleColumnConfig):
    """Configuration for columns generated using numerical samplers.

    Sampler columns provide efficient data generation using numerical samplers for
    common data types and distributions. Supported samplers include UUID generation,
    datetime/timedelta sampling, person generation, category / subcategory sampling,
    and various statistical distributions (uniform, gaussian, binomial, poisson, scipy).

    Attributes:
        sampler_type: Type of sampler to use. Available types include:
            "uuid", "category", "subcategory", "uniform", "gaussian", "bernoulli",
            "bernoulli_mixture", "binomial", "poisson", "scipy", "person", "datetime", "timedelta".
        params: Parameters specific to the chosen sampler type. Type varies based on the `sampler_type`
            (e.g., `CategorySamplerParams`, `UniformSamplerParams`, `PersonSamplerParams`).
        conditional_params: Optional dictionary for conditional parameters. The dict keys
            are the conditions that must be met (e.g., "age > 21") for the conditional parameters
            to be used. The values of dict are the parameters to use when the condition is met.
        convert_to: Optional type conversion to apply after sampling. Must be one of "float", "int", or "str".
            Useful for converting numerical samples to strings or other types.
        column_type: Discriminator field, always "sampler" for this configuration type.

    !!! tip "Displaying available samplers and their parameters"
        The config builder has an `info` attribute that can be used to display the
        available samplers and their parameters:
        ```python
        config_builder.info.display("samplers")
        ```
    """

    sampler_type: SamplerType
    params: Annotated[SamplerParamsT, Discriminator("sampler_type")]
    conditional_params: dict[str, Annotated[SamplerParamsT, Discriminator("sampler_type")]] = {}
    convert_to: Optional[str] = None
    column_type: Literal["sampler"] = "sampler"

    @model_validator(mode="before")
    @classmethod
    def inject_sampler_type_into_params(cls, data: dict) -> dict:
        """Inject sampler_type into params dict to enable discriminated union resolution.

        This allows users to pass params as a simple dict without the sampler_type field,
        which will be automatically added based on the outer sampler_type field.
        """
        if isinstance(data, dict):
            sampler_type = data.get("sampler_type")
            params = data.get("params")

            # If params is a dict and doesn't have sampler_type, inject it
            if sampler_type and isinstance(params, dict) and "sampler_type" not in params:
                data["params"] = {"sampler_type": sampler_type, **params}

            # Handle conditional_params similarly
            conditional_params = data.get("conditional_params")
            if conditional_params and isinstance(conditional_params, dict):
                for condition, cond_params in conditional_params.items():
                    if isinstance(cond_params, dict) and "sampler_type" not in cond_params:
                        data["conditional_params"][condition] = {"sampler_type": sampler_type, **cond_params}

        return data


class LLMTextColumnConfig(SingleColumnConfig):
    """Configuration for text generation columns using Large Language Models.

    LLM text columns generate free-form text content using language models via LiteLLM.
    Prompts support Jinja2 templating to reference values from other columns, enabling
    context-aware generation. The generated text can optionally include reasoning traces
    when models support extended thinking.

    Attributes:
        prompt: Prompt template for text generation. Supports Jinja2 syntax to
            reference other columns (e.g., "Write a story about {{ character_name }}").
            Must be a valid Jinja2 template.
        model_alias: Alias of the model configuration to use for generation.
            Must match a model alias defined when initializing the DataDesignerConfigBuilder.
        system_prompt: Optional system prompt to set model behavior and constraints.
            Also supports Jinja2 templating. If provided, must be a valid Jinja2 template.
            Do not put any output parsing instructions in the system prompt. Instead,
            use the appropriate column type for the output you want to generate - e.g.,
            `LLMStructuredColumnConfig` for structured output, `LLMCodeColumnConfig` for code.
        multi_modal_context: Optional list of image contexts for multi-modal generation.
            Enables vision-capable models to generate text based on image inputs.
        column_type: Discriminator field, always "llm-text" for this configuration type.
    """

    prompt: str
    model_alias: str
    system_prompt: Optional[str] = None
    multi_modal_context: Optional[list[ImageContext]] = None
    column_type: Literal["llm-text"] = "llm-text"

    @property
    def required_columns(self) -> list[str]:
        """Get columns referenced in the prompt and system_prompt templates.

        Returns:
            List of unique column names referenced in Jinja2 templates.
        """
        required_cols = list(get_prompt_template_keywords(self.prompt))
        if self.system_prompt:
            required_cols.extend(list(get_prompt_template_keywords(self.system_prompt)))
        return list(set(required_cols))

    @property
    def side_effect_columns(self) -> list[str]:
        """Returns the reasoning trace column, which may be generated alongside the main column.

        Reasoning traces are only returned if the served model parses and returns reasoning content.

        Returns:
            List containing the reasoning trace column name.
        """
        return [f"{self.name}{REASONING_TRACE_COLUMN_POSTFIX}"]

    @model_validator(mode="after")
    def assert_prompt_valid_jinja(self) -> Self:
        """Validate that prompt and system_prompt are valid Jinja2 templates.

        Returns:
            The validated instance.

        Raises:
            InvalidConfigError: If prompt or system_prompt contains invalid Jinja2 syntax.
        """
        assert_valid_jinja2_template(self.prompt)
        if self.system_prompt:
            assert_valid_jinja2_template(self.system_prompt)
        return self


class LLMCodeColumnConfig(LLMTextColumnConfig):
    """Configuration for code generation columns using Large Language Models.

    Extends LLMTextColumnConfig to generate code snippets in specific programming languages
    or SQL dialects. The generated code is automatically extracted from markdown code blocks
    for the specified language. Inherits all prompt templating capabilities.

    Attributes:
        code_lang: Programming language or SQL dialect for code generation. Supported
            values include: "python", "javascript", "typescript", "java", "kotlin", "go",
            "rust", "ruby", "scala", "swift", "sql:sqlite", "sql:postgres", "sql:mysql",
            "sql:tsql", "sql:bigquery", "sql:ansi". See CodeLang enum for complete list.
        column_type: Discriminator field, always "llm-code" for this configuration type.
    """

    code_lang: CodeLang
    column_type: Literal["llm-code"] = "llm-code"


class LLMStructuredColumnConfig(LLMTextColumnConfig):
    """Configuration for structured JSON generation columns using Large Language Models.

    Extends LLMTextColumnConfig to generate structured data conforming to a specified schema.
    Uses JSON schema or Pydantic models to define the expected output structure, enabling
    type-safe and validated structured output generation. Inherits prompt templating capabilities.

    Attributes:
        output_format: The schema defining the expected output structure. Can be either:
            - A Pydantic BaseModel class (recommended)
            - A JSON schema dictionary
        column_type: Discriminator field, always "llm-structured" for this configuration type.
    """

    output_format: Union[dict, Type[BaseModel]]
    column_type: Literal["llm-structured"] = "llm-structured"

    @model_validator(mode="after")
    def validate_output_format(self) -> Self:
        """Convert Pydantic model to JSON schema if needed.

        Returns:
            The validated instance with output_format as a JSON schema dict.
        """
        if not isinstance(self.output_format, dict) and issubclass(self.output_format, BaseModel):
            self.output_format = self.output_format.model_json_schema()
        return self


class Score(ConfigBase):
    """Configuration for a "score" in an LLM judge evaluation.

    Defines a single scoring criterion with its possible values and descriptions. Multiple
    Score objects can be combined in an LLMJudgeColumnConfig to create multi-dimensional
    quality assessments.

    Attributes:
        name: A clear, concise name for this scoring dimension (e.g., "Relevance", "Fluency").
        description: An informative and detailed assessment guide explaining how to evaluate
            this dimension. Should provide clear criteria for scoring.
        options: Dictionary mapping score values to their descriptions. Keys can be integers
            (e.g., 1-5 scale) or strings (e.g., "Poor", "Good", "Excellent"). Values are
            descriptions explaining what each score level means.
    """

    name: str = Field(..., description="A clear name for this score.")
    description: str = Field(..., description="An informative and detailed assessment guide for using this score.")
    options: dict[Union[int, str], str] = Field(..., description="Score options in the format of {score: description}.")


class LLMJudgeColumnConfig(LLMTextColumnConfig):
    """Configuration for LLM-as-a-judge quality assessment and scoring columns.

    Extends LLMTextColumnConfig to create judge columns that evaluate and score other
    generated content based on the defined criteria. Useful for quality assessment, preference
    ranking, and multi-dimensional evaluation of generated data.

    Attributes:
        scores: List of Score objects defining the evaluation dimensions. Each score
            represents a different aspect to evaluate (e.g., accuracy, relevance, fluency).
            Must contain at least one score.
        column_type: Discriminator field, always "llm-judge" for this configuration type.
    """

    scores: list[Score] = Field(..., min_length=1)
    column_type: Literal["llm-judge"] = "llm-judge"


class ExpressionColumnConfig(SingleColumnConfig):
    """Configuration for derived columns using Jinja2 expressions.

    Expression columns compute values by evaluating Jinja2 templates that reference other
    columns. Useful for transformations, concatenations, conditional logic, and derived
    features without requiring LLM generation. The expression is evaluated row-by-row.

    Attributes:
        expr: Jinja2 expression to evaluate. Can reference other column values using
            {{ column_name }} syntax. Supports filters, conditionals, and arithmetic.
            Must be a valid, non-empty Jinja2 template.
        dtype: Data type to cast the result to. Must be one of "int", "float", "str", or "bool".
            Defaults to "str". Type conversion is applied after expression evaluation.
        column_type: Discriminator field, always "expression" for this configuration type.
    """

    name: str
    expr: str
    dtype: Literal["int", "float", "str", "bool"] = "str"
    column_type: Literal["expression"] = "expression"

    @property
    def required_columns(self) -> list[str]:
        """Returns the columns referenced in the expression template."""
        return list(get_prompt_template_keywords(self.expr))

    @model_validator(mode="after")
    def assert_expression_valid_jinja(self) -> Self:
        """Validate that the expression is a valid, non-empty Jinja2 template.

        Returns:
            The validated instance.

        Raises:
            InvalidConfigError: If expression is empty or contains invalid Jinja2 syntax.
        """
        if not self.expr.strip():
            raise InvalidConfigError(
                f"🛑 Expression column '{self.name}' has an empty or whitespace-only expression. "
                f"Please provide a valid Jinja2 expression (e.g., '{{ column_name }}' or '{{ col1 }} + {{ col2 }}') "
                "or remove this column if not needed."
            )
        assert_valid_jinja2_template(self.expr)
        return self


class ValidationColumnConfig(SingleColumnConfig):
    """Configuration for validation columns that validate existing columns.

    Validation columns execute validation logic against specified target columns and return
    structured results indicating pass/fail status with validation details. Supports multiple
    validation strategies: code execution (Python/SQL), local callable functions (library only),
    and remote HTTP endpoints.

    Attributes:
        target_columns: List of column names to validate. These columns are passed to the
            validator for validation. All target columns must exist in the dataset
            before validation runs.
        validator_type: The type of validator to use. Options:
            - "code": Execute code (Python or SQL) for validation. The code receives a
              DataFrame with target columns and must return a DataFrame with validation results.
            - "local_callable": Call a local Python function with the data. Only supported
              when running DataDesigner locally.
            - "remote": Send data to a remote HTTP endpoint for validation. Useful for
        validator_params: Parameters specific to the validator type. Type varies by validator:
            - CodeValidatorParams: Specifies code language (python or SQL dialect like
              "sql:postgres", "sql:mysql").
            - LocalCallableValidatorParams: Provides validation function (Callable[[pd.DataFrame],
              pd.DataFrame]) and optional output schema for validation results.
            - RemoteValidatorParams: Configures endpoint URL, HTTP timeout, retry behavior
              (max_retries, retry_backoff), and parallel request limits (max_parallel_requests).
        batch_size: Number of records to process in each validation batch. Defaults to 10.
            Larger batches are more efficient but use more memory. Adjust based on validator
            complexity and available resources.
        column_type: Discriminator field, always "validation" for this configuration type.
    """

    target_columns: list[str]
    validator_type: ValidatorType
    validator_params: ValidatorParamsT
    batch_size: int = Field(default=10, ge=1, description="Number of records to process in each batch")
    column_type: Literal["validation"] = "validation"

    @property
    def required_columns(self) -> list[str]:
        """Returns the columns that need to be validated."""
        return self.target_columns


class SeedDatasetColumnConfig(SingleColumnConfig):
    """Configuration for columns sourced from seed datasets.

    This config marks columns that come from seed data. It is typically created
    automatically when calling `with_seed_dataset()` on the builder, rather than
    being instantiated directly by users.

    Attributes:
        column_type: Discriminator field, always "seed-dataset" for this configuration type.
    """

    column_type: Literal["seed-dataset"] = "seed-dataset"
