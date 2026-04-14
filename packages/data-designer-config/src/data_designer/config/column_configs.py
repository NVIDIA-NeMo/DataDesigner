# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field, field_serializer, field_validator, model_validator
from typing_extensions import Self

from data_designer.config.base import ConfigBase, SingleColumnConfig
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ImageContext
from data_designer.config.sampler_params import SamplerParamsT, SamplerType
from data_designer.config.utils.code_lang import CodeLang
from data_designer.config.utils.constants import REASONING_CONTENT_COLUMN_POSTFIX, TRACE_COLUMN_POSTFIX
from data_designer.config.utils.misc import assert_valid_jinja2_template, extract_keywords_from_jinja2_template
from data_designer.config.utils.trace_type import TraceType
from data_designer.config.validator_params import ValidatorParamsT, ValidatorType


class GenerationStrategy(str, Enum):
    """Strategy for custom column generation."""

    CELL_BY_CELL = "cell_by_cell"
    FULL_COLUMN = "full_column"


class SamplerColumnConfig(SingleColumnConfig):
    """Configuration for columns generated using numerical samplers.

    Sampler columns provide efficient data generation using numerical samplers for
    common data types and distributions. Supported samplers include UUID generation,
    datetime/timedelta sampling, person generation, category / subcategory sampling,
    and various statistical distributions (uniform, gaussian, binomial, poisson, scipy).

    Attributes:
        sampler_type (required): Type of sampler to use. Available types include:
            "uuid", "category", "subcategory", "uniform", "gaussian", "bernoulli",
            "bernoulli_mixture", "binomial", "poisson", "scipy", "person", "datetime", "timedelta".
        params (required): Parameters specific to the chosen sampler type. Type varies based on the `sampler_type`
            (e.g., `CategorySamplerParams`, `UniformSamplerParams`, `PersonSamplerParams`).
        conditional_params: Optional dictionary for conditional parameters. The dict keys
            are the conditions that must be met (e.g., "age > 21") for the conditional parameters
            to be used. The values of dict are the parameters to use when the condition is met.
        convert_to: Optional type conversion to apply after sampling. For numerical samplers,
            must be one of "float", "int", or "str". For datetime and timedelta samplers, accepts
            a strftime format string (e.g., ``"%Y-%m-%d"``, ``"%m/%d/%Y %H:%M"``). When omitted,
            datetime/timedelta columns default to ISO-8601 format (e.g., ``2024-01-15T09:30:00``).

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.

    !!! tip "Displaying available samplers and their parameters"
        The config builder has an `info` attribute that can be used to display the
        available samplers and their parameters:
        ```python
        config_builder.info.display("samplers")
        ```
    """

    sampler_type: SamplerType = Field(
        description="Type of sampler to use (e.g., uuid, category, uniform, gaussian, person, datetime)"
    )
    params: Annotated[SamplerParamsT, Discriminator("sampler_type")] = Field(
        description="Parameters specific to the chosen sampler type"
    )
    conditional_params: dict[str, Annotated[SamplerParamsT, Discriminator("sampler_type")]] = Field(
        default_factory=dict,
        description="Optional dictionary for conditional parameters; keys are conditions, values are params to use when met",
    )
    convert_to: str | None = Field(
        default=None,
        description=(
            "Optional type conversion after sampling: 'float', 'int', or 'str' for numerical samplers; "
            "a strftime format string (e.g., '%Y-%m-%d') for datetime/timedelta samplers. "
            "Datetime/timedelta columns default to ISO-8601 (e.g., 2024-01-15T09:30:00) when omitted."
        ),
    )
    column_type: Literal["sampler"] = "sampler"

    @staticmethod
    def get_column_emoji() -> str:
        return "🎲"

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []

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

    LLM text columns generate free-form text content using language models.
    Prompts support Jinja2 templating to reference values from other columns, enabling
    context-aware generation. The generated text can optionally include message traces
    capturing the full conversation history.

    Attributes:
        prompt (required): Prompt template for text generation. Supports Jinja2 syntax to
            reference other columns (e.g., "Write a story about {{ character_name }}").
            Must be a valid Jinja2 template.
        model_alias (required): Alias of the model configuration to use for generation.
            Must match a model alias defined when initializing the DataDesignerConfigBuilder.
        system_prompt: Optional system prompt to set model behavior and constraints.
            Also supports Jinja2 templating. If provided, must be a valid Jinja2 template.
            Do not put any output parsing instructions in the system prompt. Instead,
            use the appropriate column type for the output you want to generate - e.g.,
            `LLMStructuredColumnConfig` for structured output, `LLMCodeColumnConfig` for code.
        multi_modal_context: Optional list of image contexts for multi-modal generation.
            Enables vision-capable models to generate text based on image inputs.
        tool_alias: Optional alias of the tool configuration to use for MCP tool calls.
            Must match a tool alias defined when initializing the DataDesignerConfigBuilder.
            When provided, the model may call permitted tools during generation.
        with_trace: Specifies what trace information to capture in a `{column_name}__trace`
            column. Options are:
            - `TraceType.NONE` (default): No trace is captured.
            - `TraceType.LAST_MESSAGE`: Only the final assistant message is captured.
            - `TraceType.ALL_MESSAGES`: Full conversation history (system/user/assistant/tool).
        extract_reasoning_content: If True, creates a `{column_name}__reasoning_content` column
            containing only the reasoning_content from the final assistant response. This is
            useful for models that expose chain-of-thought reasoning separately from the main
            response. Defaults to False.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    prompt: str = Field(
        description="Jinja2 template for the LLM prompt; can reference other columns via {{ column_name }}"
    )
    model_alias: str = Field(description="Alias of the model configuration to use for generation")
    system_prompt: str | None = Field(
        default=None, description="Optional system prompt to set model behavior and constraints"
    )
    multi_modal_context: list[ImageContext] | None = Field(
        default=None, description="Optional list of ImageContext for vision model inputs"
    )
    tool_alias: str | None = Field(
        default=None, description="Optional alias of the tool configuration to use for MCP tool calls"
    )
    with_trace: TraceType = Field(
        default=TraceType.NONE, description="Trace capture mode: NONE, LAST_MESSAGE, or ALL_MESSAGES"
    )
    extract_reasoning_content: bool = Field(
        default=False, description="If True, capture chain-of-thought in {name}__reasoning_content column"
    )
    column_type: Literal["llm-text"] = "llm-text"

    @staticmethod
    def get_column_emoji() -> str:
        return "📝"

    @property
    def required_columns(self) -> list[str]:
        """Get columns referenced in prompt templates and multi-modal context.

        Returns:
            List of unique column names referenced in Jinja2 templates
            and multi-modal context configurations.
        """
        required_cols = list(extract_keywords_from_jinja2_template(self.prompt))
        if self.system_prompt:
            required_cols.extend(list(extract_keywords_from_jinja2_template(self.system_prompt)))
        if self.multi_modal_context:
            required_cols.extend(ctx.column_name for ctx in self.multi_modal_context)
        return list(set(required_cols))

    @property
    def side_effect_columns(self) -> list[str]:
        """Returns side-effect columns that may be generated alongside the main column.

        Side-effect columns include:
        - `{name}__trace`: Generated when `with_trace` is not `TraceType.NONE` on the column
          config.
        - `{name}__reasoning_content`: Generated when `extract_reasoning_content=True`.

        Returns:
            List of side-effect column names.
        """
        return [
            *([f"{self.name}{TRACE_COLUMN_POSTFIX}"] if self.with_trace != TraceType.NONE else []),
            *([f"{self.name}{REASONING_CONTENT_COLUMN_POSTFIX}"] if self.extract_reasoning_content else []),
        ]

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
    for the specified language. Inherits all prompt templating capabilities from LLMTextColumnConfig.

    Attributes:
        code_lang (required): Programming language or SQL dialect for code generation. Supported
            values include: "python", "javascript", "typescript", "java", "kotlin", "go",
            "rust", "ruby", "scala", "swift", "sql:sqlite", "sql:postgres", "sql:mysql",
            "sql:tsql", "sql:bigquery", "sql:ansi". See CodeLang enum for complete list.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        prompt (required): Prompt template for code generation (supports Jinja2).
        model_alias (required): Alias of the model configuration to use.
        system_prompt: Optional system prompt (supports Jinja2).
        multi_modal_context: Optional image contexts for multi-modal generation.
        tool_alias: Optional tool configuration alias for MCP tool calls.
        with_trace: Specifies what trace information to capture in a `{column_name}__trace`
            column. Options are `TraceType.NONE` (default), `TraceType.LAST_MESSAGE`, or
            `TraceType.ALL_MESSAGES`.
        extract_reasoning_content: If True, creates a `{column_name}__reasoning_content`
            column containing the reasoning content from the final assistant response.
        drop: If True, generate this column but remove it from the final dataset.
    """

    code_lang: CodeLang = Field(
        description="Target programming language or SQL dialect for code extraction from LLM response"
    )
    column_type: Literal["llm-code"] = "llm-code"

    @staticmethod
    def get_column_emoji() -> str:
        return "💻"


class LLMStructuredColumnConfig(LLMTextColumnConfig):
    """Configuration for structured JSON generation columns using Large Language Models.

    Extends LLMTextColumnConfig to generate structured data conforming to a specified schema.
    Uses JSON schema or Pydantic models to define the expected output structure, enabling
    type-safe and validated structured output generation. Inherits prompt templating capabilities
    from LLMTextColumnConfig.

    Attributes:
        output_format (required): The schema defining the expected output structure. Can be either:
            - A Pydantic BaseModel class (recommended)
            - A JSON schema dictionary

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        prompt (required): Prompt template for structured generation (supports Jinja2).
        model_alias (required): Alias of the model configuration to use.
        system_prompt: Optional system prompt (supports Jinja2).
        multi_modal_context: Optional image contexts for multi-modal generation.
        tool_alias: Optional tool configuration alias for MCP tool calls.
        with_trace: Specifies what trace information to capture in a `{column_name}__trace`
            column. Options are `TraceType.NONE` (default), `TraceType.LAST_MESSAGE`, or
            `TraceType.ALL_MESSAGES`.
        extract_reasoning_content: If True, creates a `{column_name}__reasoning_content`
            column containing the reasoning content from the final assistant response.
        drop: If True, generate this column but remove it from the final dataset.
    """

    output_format: dict | type[BaseModel] = Field(
        description="Pydantic model or JSON schema dict defining the expected structured output shape"
    )
    column_type: Literal["llm-structured"] = "llm-structured"

    @staticmethod
    def get_column_emoji() -> str:
        return "🗂️"

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
        name (required): A clear, concise name for this scoring dimension (e.g., "Relevance", "Fluency").
        description (required): An informative and detailed assessment guide explaining how to evaluate
            this dimension. Should provide clear criteria for scoring.
        options (required): Dictionary mapping score values to their descriptions. Keys can be integers
            (e.g., 1-5 scale) or strings (e.g., "Poor", "Good", "Excellent"). Values are
            descriptions explaining what each score level means.
    """

    name: str = Field(..., description="A clear name for this score.")
    description: str = Field(..., description="An informative and detailed assessment guide for using this score.")
    options: dict[int | str, str] = Field(..., description="Score options in the format of {score: description}.")


class LLMJudgeColumnConfig(LLMTextColumnConfig):
    """Configuration for LLM-as-a-judge quality assessment and scoring columns.

    Extends LLMTextColumnConfig to create judge columns that evaluate and score other
    generated content based on the defined criteria. Useful for quality assessment, preference
    ranking, and multi-dimensional evaluation of generated data. Inherits prompt templating
    capabilities from LLMTextColumnConfig.

    Attributes:
        scores (required): List of Score objects defining the evaluation dimensions. Each score
            represents a different aspect to evaluate (e.g., accuracy, relevance, fluency).
            Must contain at least one score.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        prompt (required): Prompt template for the judge evaluation (supports Jinja2).
        model_alias (required): Alias of the model configuration to use.
        system_prompt: Optional system prompt (supports Jinja2).
        multi_modal_context: Optional image contexts for multi-modal generation.
        tool_alias: Optional tool configuration alias for MCP tool calls.
        with_trace: Specifies what trace information to capture in a `{column_name}__trace`
            column. Options are `TraceType.NONE` (default), `TraceType.LAST_MESSAGE`, or
            `TraceType.ALL_MESSAGES`.
        extract_reasoning_content: If True, creates a `{column_name}__reasoning_content`
            column containing the reasoning content from the final assistant response.
        drop: If True, generate this column but remove it from the final dataset.
    """

    scores: list[Score] = Field(
        ..., min_length=1, description="List of Score objects defining rubric criteria for LLM judge evaluation"
    )
    column_type: Literal["llm-judge"] = "llm-judge"

    @staticmethod
    def get_column_emoji() -> str:
        return "⚖️"


class ExpressionColumnConfig(SingleColumnConfig):
    """Configuration for derived columns using Jinja2 expressions.

    Expression columns compute values by evaluating Jinja2 templates that reference other
    columns. Useful for transformations, concatenations, conditional logic, and derived
    features without requiring LLM generation. The expression is evaluated row-by-row.

    Attributes:
        expr (required): Jinja2 expression to evaluate. Can reference other column values using
            {{ column_name }} syntax. Supports filters, conditionals, and arithmetic.
            Must be a valid, non-empty Jinja2 template.
        dtype: Data type to cast the result to. Must be one of "int", "float", "str", or "bool".
            Defaults to "str". Type conversion is applied after expression evaluation.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    expr: str = Field(description="Jinja2 expression to compute the column value from other columns")
    dtype: Literal["int", "float", "str", "bool"] = Field(
        default="str", description="Data type for expression result: 'int', 'float', 'str', or 'bool'"
    )
    column_type: Literal["expression"] = "expression"

    @staticmethod
    def get_column_emoji() -> str:
        return "🧩"

    @property
    def required_columns(self) -> list[str]:
        """Returns the columns referenced in the expression template."""
        return list(extract_keywords_from_jinja2_template(self.expr))

    @property
    def side_effect_columns(self) -> list[str]:
        return []

    _DTYPE_COERCERS: dict[str, type] = {
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
    }

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

    @model_validator(mode="after")
    def _coerce_skip_value_to_dtype(self) -> Self:
        """Coerce ``skip.value`` to match ``dtype`` so skipped and computed rows share a type."""
        if self.skip is None or self.skip.value is None:
            return self
        target_type = self._DTYPE_COERCERS.get(self.dtype)
        if target_type is not None and not isinstance(self.skip.value, target_type):
            try:
                self.skip.value = target_type(self.skip.value)
            except (ValueError, TypeError) as exc:
                raise ValueError(
                    f"Expression column '{self.name}' has dtype='{self.dtype}' but "
                    f"skip.value={self.skip.value!r} cannot be converted to {self.dtype}: {exc}"
                ) from exc
        return self


class ValidationColumnConfig(SingleColumnConfig):
    """Configuration for validation columns that validate existing columns.

    Validation columns execute validation logic against specified target columns and return
    structured results indicating pass/fail status with validation details. Supports multiple
    validation strategies: code execution (Python/SQL), local callable functions (library only),
    and remote HTTP endpoints.

    Attributes:
        target_columns (required): List of column names to validate. These columns are passed to the
            validator for validation. All target columns must exist in the dataset
            before validation runs.
        validator_type (required): The type of validator to use. Options:
            - "code": Execute code (Python or SQL) for validation. The code receives a
              DataFrame with target columns and must return a DataFrame with validation results.
            - "local_callable": Call a local Python function with the data. Only supported
              when running DataDesigner locally.
            - "remote": Send data to a remote HTTP endpoint for validation. Useful for
        validator_params (required): Parameters specific to the validator type. Type varies by validator:
            - CodeValidatorParams: Specifies code language (python or SQL dialect like
              "sql:postgres", "sql:mysql").
            - LocalCallableValidatorParams: Provides validation function (Callable[[pd.DataFrame],
              pd.DataFrame]) and optional output schema for validation results.
            - RemoteValidatorParams: Configures endpoint URL, HTTP timeout, retry behavior
              (max_retries, retry_backoff), and parallel request limits (max_parallel_requests).
        batch_size: Number of records to process in each validation batch. Defaults to 10.
            Larger batches are more efficient but use more memory. Adjust based on validator
            complexity and available resources.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    target_columns: list[str] = Field(description="List of column names to validate")
    validator_type: ValidatorType = Field(description="Validation method: 'code', 'local_callable', or 'remote'")
    validator_params: Annotated[ValidatorParamsT, Discriminator("validator_type")] = Field(
        description="Validator-specific parameters (e.g., CodeValidatorParams)"
    )
    batch_size: int = Field(default=10, ge=1, description="Number of records to process in each batch")
    column_type: Literal["validation"] = "validation"

    @staticmethod
    def get_column_emoji() -> str:
        return "🔍"

    @property
    def required_columns(self) -> list[str]:
        """Returns the columns that need to be validated."""
        return self.target_columns

    @property
    def side_effect_columns(self) -> list[str]:
        return []

    @model_validator(mode="before")
    @classmethod
    def inject_validator_type_into_params(cls, data: dict) -> dict:
        """Inject validator_type into validator_params for discriminated union resolution."""
        if isinstance(data, dict):
            validator_type = data.get("validator_type")
            validator_params = data.get("validator_params")
            if validator_type and isinstance(validator_params, dict) and "validator_type" not in validator_params:
                data["validator_params"] = {"validator_type": validator_type, **validator_params}
        return data


class SeedDatasetColumnConfig(SingleColumnConfig):
    """Configuration for columns sourced from seed datasets.

    This config marks columns that come from seed data. It is typically created
    automatically when calling `with_seed_dataset()` on the builder, rather than
    being instantiated directly by users.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    column_type: Literal["seed-dataset"] = "seed-dataset"

    @staticmethod
    def get_column_emoji() -> str:
        return "🌱"

    @property
    def required_columns(self) -> list[str]:
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class EmbeddingColumnConfig(SingleColumnConfig):
    """Configuration for embedding generation columns.

    Embedding columns generate embeddings for text input using a specified model.

    Attributes:
        target_column (required): The column to generate embeddings for. The column could be a single text string or a list of text strings in stringified JSON format.
            If it is a list of text strings in stringified JSON format, the embeddings will be generated for each text string.
        model_alias (required): The model to use for embedding generation.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    target_column: str = Field(description="Name of the text column to generate embeddings for")
    model_alias: str = Field(description="Alias of the model to use for embedding generation")
    column_type: Literal["embedding"] = "embedding"

    @staticmethod
    def get_column_emoji() -> str:
        return "🧬"

    @property
    def required_columns(self) -> list[str]:
        return [self.target_column]

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class ImageColumnConfig(SingleColumnConfig):
    """Configuration for image generation columns.

    Image columns generate images using either autoregressive or diffusion models.
    The API used is automatically determined based on the model name:

    Attributes:
        prompt (required): Prompt template for image generation. Supports Jinja2 templating to
            reference other columns (e.g., "Generate an image of a {{ character_name }}").
            Must be a valid Jinja2 template.
        model_alias (required): The model to use for image generation.
        multi_modal_context: Optional list of image contexts for multi-modal generation.
            Enables autoregressive multi-modal models to generate images based on image inputs.
            Only works with autoregressive models that support image-to-image generation.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    prompt: str = Field(
        description="Jinja2 template for the image generation prompt; can reference other columns via {{ column_name }}"
    )
    model_alias: str = Field(description="Alias of the model to use for image generation")
    multi_modal_context: list[ImageContext] | None = Field(
        default=None, description="Optional list of ImageContext for multi-modal image-to-image generation"
    )
    column_type: Literal["image"] = "image"

    @staticmethod
    def get_column_emoji() -> str:
        return "🖼️"

    @property
    def required_columns(self) -> list[str]:
        """Get columns referenced in the prompt template and multi-modal context.

        Returns:
            List of unique column names referenced in Jinja2 templates
            and multi-modal context configurations.
        """
        required_cols = list(extract_keywords_from_jinja2_template(self.prompt))
        if self.multi_modal_context:
            required_cols.extend(ctx.column_name for ctx in self.multi_modal_context)
        return list(set(required_cols))

    @model_validator(mode="after")
    def assert_prompt_valid_jinja(self) -> Self:
        """Validate that prompt is a valid Jinja2 template.

        Returns:
            The validated instance.

        Raises:
            InvalidConfigError: If prompt contains invalid Jinja2 syntax.
        """
        assert_valid_jinja2_template(self.prompt)
        return self

    @property
    def side_effect_columns(self) -> list[str]:
        return []


class CustomColumnConfig(SingleColumnConfig):
    """Configuration for custom user-defined column generators.

    Custom columns allow users to provide their own generation logic via a callable function
    decorated with `@custom_column_generator`. Two strategies are supported: cell_by_cell
    (default, row-based) and full_column (batch-based with DataFrame access).

    Attributes:
        generator_function (required): A callable decorated with @custom_column_generator.
        generation_strategy: "cell_by_cell" (row-based) or "full_column" (batch-based).
        generator_params: Optional typed configuration object (Pydantic BaseModel) passed
            as the second argument to the generator function.

    Inherited Attributes:
        name (required): Unique name of the column to be generated.
        drop: If True, generate this column but remove it from the final dataset.
    """

    generator_function: Any = Field(description="Function decorated with @custom_column_generator")
    generation_strategy: GenerationStrategy = Field(
        default=GenerationStrategy.CELL_BY_CELL,
        description="Generation strategy: 'cell_by_cell' for row-based or 'full_column' for batch-based",
    )
    generator_params: BaseModel | None = Field(
        default=None,
        description="Optional typed configuration object passed as second argument to generator function",
    )
    column_type: Literal["custom"] = "custom"

    @field_validator("generator_function")
    @classmethod
    def _validate_generator_function(cls, v: Any) -> Any:
        if not callable(v):
            raise ValueError("generator_function must be callable")
        if not hasattr(v, "custom_column_metadata"):
            raise ValueError("generator_function must be decorated with @custom_column_generator")
        return v

    @staticmethod
    def get_column_emoji() -> str:
        return "🔧"

    @property
    def required_columns(self) -> list[str]:
        """Returns the columns required for custom generation (from decorator metadata)."""
        metadata = getattr(self.generator_function, "custom_column_metadata", {})
        return metadata.get("required_columns", [])

    @property
    def side_effect_columns(self) -> list[str]:
        """Returns additional columns created by this generator (from decorator metadata)."""
        metadata = getattr(self.generator_function, "custom_column_metadata", {})
        return metadata.get("side_effect_columns", [])

    @property
    def model_aliases(self) -> list[str]:
        """Returns model aliases for LLM access and health checks (from decorator metadata)."""
        metadata = getattr(self.generator_function, "custom_column_metadata", {})
        return metadata.get("model_aliases", [])

    @field_serializer("generator_function")
    def serialize_generator_function(self, v: Any) -> str:
        return getattr(v, "__name__", repr(v))

    @field_serializer("generator_params")
    def serialize_generator_params(self, v: BaseModel | None) -> dict[str, Any] | None:
        if v is None:
            return None
        return v.model_dump()

    @model_validator(mode="after")
    def validate_generator_function(self) -> Self:
        if not callable(self.generator_function):
            raise InvalidConfigError(
                f"🛑 `generator_function` must be a callable for custom column '{self.name}'. "
                f"Expected a function decorated with @custom_column_generator."
            )
        return self
