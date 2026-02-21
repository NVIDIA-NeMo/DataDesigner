"""Print LLM-optimized context for data designer column types.

Usage:
    uv run get_column_info.py              # Print help
    uv run get_column_info.py all          # Print all column types
    uv run get_column_info.py llm-text     # Print specific column type
    uv run get_column_info.py --list       # List column types and config classes
"""

import inspect
from typing import Literal, get_args, get_origin

from helpers.pydantic_info_utils import run_cli

import data_designer.config as dd

# Default descriptions for common fields that lack docstrings.
# These are derived from inspecting how fields are used in the codebase.
DEFAULT_FIELD_DESCRIPTIONS: dict[str, str] = {
    # Common base fields
    "name": "Unique column name in the generated dataset",
    "drop": "If True, exclude this column from the final dataset output",
    "column_type": "Discriminator identifying the column configuration type",
    # LLM-related fields
    "prompt": "Jinja2 template for the LLM prompt; can reference other columns via {{ column_name }}",
    "model_alias": "Reference to a ModelConfig.alias defined in the config builder",
    "system_prompt": "Optional system prompt to set LLM behavior and context",
    "multi_modal_context": "Optional list of ImageContext for vision model inputs",
    "tool_alias": "Optional reference to a ToolConfig.tool_alias for MCP tool access",
    "with_trace": "Trace capture mode: NONE, LAST_MESSAGE, or ALL_MESSAGES",
    "extract_reasoning_content": "If True, capture chain-of-thought in {name}__reasoning_content column",
    # LLM type-specific fields
    "code_lang": "Target programming language for code extraction from LLM response",
    "scores": "List of Score objects defining rubric criteria for LLM judge evaluation",
    "output_format": "Pydantic model or JSON schema dict defining the structured output shape",
    # Sampler fields
    "sampler_type": "Type of statistical sampler to use (e.g., CATEGORY, UNIFORM, PERSON)",
    "params": "Sampler-specific parameters (e.g., CategorySamplerParams, UniformSamplerParams)",
    "conditional_params": "Override params based on conditions referencing other columns",
    "convert_to": "Optional type cast for sampled values: 'int', 'float', or 'str'",
    # Expression fields
    "expr": "Jinja2 expression to compute the column value from other columns",
    "dtype": "Data type for expression result: 'int', 'float', 'str', or 'bool'",
    # Embedding fields
    "target_column": "Name of the text column to generate embeddings for",
    # Validation fields
    "target_columns": "List of column names to validate",
    "validator_type": "Validation method: CODE, LOCAL_CALLABLE, or REMOTE",
    "validator_params": "Validator-specific parameters (e.g., CodeValidatorParams)",
}


def discover_column_configs() -> dict[str, type]:
    """Dynamically discover all ColumnConfig classes from data_designer.config.

    Returns:
        Dict mapping column_type values (e.g., 'llm-text') to their config classes.
    """
    column_configs = {}
    for name in dir(dd):
        if name.endswith("ColumnConfig"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                if "column_type" in obj.model_fields:
                    annotation = obj.model_fields["column_type"].annotation
                    if get_origin(annotation) is Literal:
                        args = get_args(annotation)
                        if args:
                            column_configs[args[0]] = obj
    return column_configs


def main() -> None:
    """CLI entry point."""
    run_cli(
        discover_fn=discover_column_configs,
        type_key="column_type",
        type_label="column_type",
        class_label="config_class",
        default_descriptions=DEFAULT_FIELD_DESCRIPTIONS,
        script_name="get_column_info.py",
        description="Print LLM-optimized context for data designer column types.",
        header_title="Data Designer Column Types Reference",
        examples=[
            "uv run get_column_info.py all       # Print all column types",
            "uv run get_column_info.py llm-text  # Print specific column type",
            "uv run get_column_info.py --list    # List column types and config classes",
        ],
        case_insensitive=False,
        uppercase_value=False,
    )


if __name__ == "__main__":
    main()
