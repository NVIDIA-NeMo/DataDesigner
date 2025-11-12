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
    """Abstract base class for all single-column configuration types.

    This class serves as the foundation for all column configurations in DataDesigner,
    defining the common interface and attributes that all column types must implement.

    Attributes:
        name: The unique name of the column to be generated.
        drop: If True, the column will be generated but removed from the final dataset.
            Useful for intermediate columns that are dependencies for other columns.
        column_type: A discriminator field that identifies the specific column type.
            Used for polymorphic deserialization of column configurations.
    """

    name: str
    drop: bool = False
    column_type: str

    @property
    def required_columns(self) -> list[str]:
        """Get the list of column names that must exist before this column can be generated.

        Returns:
            A list of column names that this column depends on. Empty list indicates
            no dependencies. Override in subclasses to specify dependencies.
        """
        return []

    @property
    def side_effect_columns(self) -> list[str]:
        """Get the list of additional columns that will be created as side effects.

        Some column types generate additional metadata or auxiliary columns alongside
        the primary column (e.g., reasoning traces for LLM columns).

        Returns:
            A list of column names that will be generated as side effects. Empty list
            indicates no side effect columns. Override in subclasses to specify side effects.
        """
        return []


class SamplerColumnConfig(SingleColumnConfig):
    """Configuration for columns generated using built-in samplers and distributions.

    Sampler columns provide efficient data generation using pre-built generators for
    common data types and distributions. Supported samplers include UUID generation,
    datetime/timedelta sampling, person/entity generation, categorical sampling,
    and various statistical distributions (uniform, gaussian, binomial, poisson, scipy).

    Attributes:
        sampler_type: The type of sampler to use. Available types include:
            "uuid", "category", "subcategory", "uniform", "gaussian", "bernoulli",
            "bernoulli_mixture", "binomial", "poisson", "scipy", "person", "datetime", "timedelta".
        params: Parameters specific to the chosen sampler type. Type varies based on
            sampler_type (e.g., CategorySamplerParams, UniformSamplerParams, PersonSamplerParams).
        conditional_params: Optional mapping for conditional sampling. Keys are condition
            values from other columns, values are parameter objects to use when that
            condition is met. Enables different sampling behavior based on other columns.
        convert_to: Optional type conversion to apply after sampling (e.g., "str", "int", "float").
            Useful for converting numerical samples to strings or other types.
        column_type: Discriminator field, always "sampler" for this configuration type.

    Example:
        ```python
        # Generate categorical colors
        SamplerColumnConfig(
            name="color",
            sampler_type="category",
            params=CategorySamplerParams(
                values=["red", "blue", "green"],
                weights=[0.5, 0.3, 0.2]
            )
        )

        # Generate synthetic people
        SamplerColumnConfig(
            name="person",
            sampler_type="person",
            params=PersonSamplerParams(
                locale="en_US",
                age_range=[25, 65],
                state=["CA", "NY"]
            )
        )
        ```
    """

    sampler_type: SamplerType
    params: SamplerParamsT
    conditional_params: dict[str, SamplerParamsT] = {}
    convert_to: Optional[str] = None
    column_type: Literal["sampler"] = "sampler"


class LLMTextColumnConfig(SingleColumnConfig):
    """Configuration for text generation columns using Large Language Models.

    LLM text columns generate free-form text content using language models via LiteLLM.
    Prompts support Jinja2 templating to reference values from other columns, enabling
    context-aware generation. The generated text can optionally include reasoning traces
    when models support extended thinking.

    Attributes:
        prompt: The prompt template for text generation. Supports Jinja2 syntax to
            reference other columns (e.g., "Write a story about {{ character_name }}").
            Must be a valid Jinja2 template.
        model_alias: The alias of the model configuration to use for generation.
            Must match a model alias defined in the DataDesigner configuration.
        system_prompt: Optional system-level prompt to set model behavior and constraints.
            Also supports Jinja2 templating. If provided, must be a valid Jinja2 template.
        multi_modal_context: Optional list of image contexts for multi-modal generation.
            Enables vision-capable models to generate text based on image inputs.
        column_type: Discriminator field, always "llm-text" for this configuration type.

    Example:
        ```python
        LLMTextColumnConfig(
            name="product_description",
            prompt="Write a compelling product description for: {{ product_name }}",
            model_alias="gpt4",
            system_prompt="You are an expert marketing copywriter."
        )
        ```
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
        """Get the reasoning trace column generated alongside the main column.

        Returns:
            List containing the reasoning trace column name, which captures the model's
            thinking process when extended thinking is enabled.
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
    and validated for the specified language. Inherits all prompt templating capabilities.

    Attributes:
        code_lang: The programming language or SQL dialect for code generation. Supported
            values include: "python", "javascript", "typescript", "java", "kotlin", "go",
            "rust", "ruby", "scala", "swift", "sql:sqlite", "sql:postgres", "sql:mysql",
            "sql:tsql", "sql:bigquery", "sql:ansi". See CodeLang enum for complete list.
        column_type: Discriminator field, always "llm-code" for this configuration type.

    Example:
        ```python
        LLMCodeColumnConfig(
            name="solution_code",
            prompt="Write a Python function to {{ task_description }}",
            model_alias="claude-sonnet",
            code_lang="python",
            system_prompt="You are an expert Python developer."
        )
        ```
    """

    code_lang: CodeLang
    column_type: Literal["llm-code"] = "llm-code"


class LLMStructuredColumnConfig(LLMTextColumnConfig):
    """Configuration for structured JSON generation columns using Large Language Models.

    Extends LLMTextColumnConfig to generate structured data conforming to a specified schema.
    Uses JSON schema or Pydantic models to define the expected output structure, enabling
    type-safe and validated structured data generation. Inherits prompt templating capabilities.

    Attributes:
        output_format: The schema defining the expected output structure. Can be either:
            - A JSON schema dictionary with keys like "type", "properties", "required"
            - A Pydantic BaseModel class (automatically converted to JSON schema)
        column_type: Discriminator field, always "llm-structured" for this configuration type.

    Example:
        ```python
        # Using Pydantic model
        from pydantic import BaseModel

        class PersonInfo(BaseModel):
            age: int
            occupation: str
            hobbies: list[str]

        LLMStructuredColumnConfig(
            name="person_details",
            prompt="Generate details for {{ name }}",
            model_alias="gpt4",
            output_format=PersonInfo
        )

        # Using JSON schema dict
        LLMStructuredColumnConfig(
            name="product_info",
            prompt="Generate product details",
            model_alias="gpt4",
            output_format={
                "type": "object",
                "properties": {
                    "price": {"type": "number"},
                    "category": {"type": "string"}
                },
                "required": ["price", "category"]
            }
        )
        ```
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
    """Configuration for a scoring dimension in LLM judge evaluations.

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

    Example:
        ```python
        Score(
            name="Accuracy",
            description="Evaluate the factual correctness of the response",
            options={
                1: "Contains multiple factual errors",
                2: "Contains minor factual errors",
                3: "Mostly accurate with small issues",
                4: "Accurate with no errors",
                5: "Perfectly accurate with excellent detail"
            }
        )
        ```
    """

    name: str = Field(..., description="A clear name for this score.")
    description: str = Field(..., description="An informative and detailed assessment guide for using this score.")
    options: dict[Union[int, str], str] = Field(..., description="Score options in the format of {score: description}.")


class LLMJudgeColumnConfig(LLMTextColumnConfig):
    """Configuration for LLM-based quality assessment and scoring columns.

    Extends LLMTextColumnConfig to create judge columns that evaluate and score other
    generated content based on defined criteria. Useful for quality assessment, preference
    ranking, and multi-dimensional evaluation of generated data. Each score dimension
    produces a separate sub-column in the output.

    Attributes:
        scores: List of Score objects defining the evaluation dimensions. Each score
            represents a different aspect to evaluate (e.g., accuracy, relevance, fluency).
            Must contain at least one score. Each score generates a separate column.
        column_type: Discriminator field, always "llm-judge" for this configuration type.

    Example:
        ```python
        LLMJudgeColumnConfig(
            name="response_quality",
            prompt="Evaluate the quality of this response: {{ generated_text }}",
            model_alias="gpt4",
            scores=[
                Score(
                    name="relevance",
                    description="How relevant is the response to the question?",
                    options={1: "Not relevant", 2: "Somewhat relevant", 3: "Very relevant"}
                ),
                Score(
                    name="clarity",
                    description="How clear and well-written is the response?",
                    options={1: "Unclear", 2: "Acceptable", 3: "Very clear"}
                )
            ]
        )
        ```
    """

    scores: list[Score] = Field(..., min_length=1)
    column_type: Literal["llm-judge"] = "llm-judge"


class ExpressionColumnConfig(SingleColumnConfig):
    """Configuration for derived columns using Jinja2 expressions.

    Expression columns compute values by evaluating Jinja2 templates that reference other
    columns. Useful for transformations, concatenations, conditional logic, and derived
    features without requiring LLM generation. The expression is evaluated row-by-row.

    Attributes:
        name: The unique name of the derived column.
        expr: The Jinja2 expression to evaluate. Can reference other column values using
            {{ column_name }} syntax. Supports filters, conditionals, and arithmetic.
            Must be a valid, non-empty Jinja2 template.
        dtype: The data type to cast the result to. One of "int", "float", "str", or "bool".
            Defaults to "str". Type conversion is applied after expression evaluation.
        column_type: Discriminator field, always "expression" for this configuration type.

    Example:
        ```python
        # Simple concatenation
        ExpressionColumnConfig(
            name="full_name",
            expr="{{ first_name }} {{ last_name }}",
            dtype="str"
        )

        # Arithmetic expression
        ExpressionColumnConfig(
            name="total_price",
            expr="{{ quantity }} * {{ unit_price }}",
            dtype="float"
        )

        # Conditional logic
        ExpressionColumnConfig(
            name="discount_tier",
            expr="{% if total > 1000 %}premium{% elif total > 500 %}standard{% else %}basic{% endif %}",
            dtype="str"
        )
        ```
    """

    name: str
    expr: str
    dtype: Literal["int", "float", "str", "bool"] = "str"
    column_type: Literal["expression"] = "expression"

    @property
    def required_columns(self) -> list[str]:
        """Get columns referenced in the expression template.

        Returns:
            List of column names referenced in the Jinja2 expression.
        """
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
    """Configuration for validation columns that check data quality and correctness.

    Validation columns execute validation logic against specified target columns and return
    structured results indicating pass/fail status and validation details. Supports multiple
    validation strategies including code execution (Python/SQL), local callable functions,
    and remote HTTP endpoints.

    Attributes:
        target_columns: List of column names to validate. These columns will be passed to
            the validator for quality assessment.
        validator_type: The type of validator to use. Options:
            - "code": Execute code (Python or SQL) for validation
            - "local_callable": Call a local Python function with the data
            - "remote": Send data to a remote HTTP endpoint for validation
        validator_params: Parameters specific to the validator type. Type varies:
            - CodeValidatorParams: Specifies code language (python or SQL dialect)
            - LocalCallableValidatorParams: Provides validation function and output schema
            - RemoteValidatorParams: Configures endpoint URL, timeout, retries, parallelism
        batch_size: Number of records to process in each validation batch. Defaults to 10.
            Larger batches may be more efficient but use more memory.
        column_type: Discriminator field, always "validation" for this configuration type.

    Example:
        ```python
        # Code validator (Python)
        ValidationColumnConfig(
            name="email_validation",
            target_columns=["email"],
            validator_type="code",
            validator_params=CodeValidatorParams(code_lang="python"),
            batch_size=50
        )

        # Remote validator
        ValidationColumnConfig(
            name="content_moderation",
            target_columns=["user_text"],
            validator_type="remote",
            validator_params=RemoteValidatorParams(
                endpoint_url="https://api.example.com/validate",
                timeout=30.0,
                max_retries=3
            )
        )
        ```
    """

    target_columns: list[str]
    validator_type: ValidatorType
    validator_params: ValidatorParamsT
    batch_size: int = Field(default=10, ge=1, description="Number of records to process in each batch")
    column_type: Literal["validation"] = "validation"

    @property
    def required_columns(self) -> list[str]:
        """Get the columns that need to be validated.

        Returns:
            List of target column names that must exist before validation runs.
        """
        return self.target_columns


class SeedDatasetColumnConfig(SingleColumnConfig):
    """Configuration for columns sourced from seed datasets.

    Seed dataset columns pull data from existing datasets provided during DataDesigner
    initialization. This enables generation workflows that start from real data and
    augment or transform it with additional synthetic columns. The seed dataset is
    specified at the DataDesigner level, and columns are referenced by name.

    Attributes:
        column_type: Discriminator field, always "seed-dataset" for this configuration type.

    Note:
        The actual seed dataset and column mapping is specified when creating the
        DataDesigner instance, not in this column configuration. This config simply
        marks that the column comes from the seed data.

    Example:
        ```python
        # Assuming seed_df contains columns: "user_id", "age", "country"
        # Mark "user_id" as coming from seed dataset
        SeedDatasetColumnConfig(
            name="user_id"
        )

        # Then add synthetic columns that reference the seed column
        LLMTextColumnConfig(
            name="user_description",
            prompt="Describe a user who is {{ age }} years old from {{ country }}",
            model_alias="gpt4"
        )
        ```
    """

    column_type: Literal["seed-dataset"] = "seed-dataset"
