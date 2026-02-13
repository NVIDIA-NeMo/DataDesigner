"""Print LLM-optimized context for data designer validator types.

Usage:
    uv run get_validator_info.py              # Print help
    uv run get_validator_info.py all          # Print all validator types
    uv run get_validator_info.py code         # Print specific validator type
    uv run get_validator_info.py --list       # List validator types and params classes
"""

import inspect
import sys
from enum import Enum

from helpers.pydantic_info_utils import run_cli

import data_designer.config as dd

DEFAULT_FIELD_DESCRIPTIONS: dict[str, str] = {
    # Code validator
    "code_lang": "Programming language for linting (e.g., PYTHON, SQL_POSTGRES, SQL_ANSI)",
    # Local callable validator
    "validation_function": "Callable[[pd.DataFrame], pd.DataFrame] that must return a df with 'is_valid' bool column",
    "output_schema": "Optional JSON schema dict for validating the output DataFrame structure",
    # Remote validator
    "endpoint_url": "HTTP endpoint URL that accepts POST with {'data': [...]} payload",
    "timeout": "Request timeout in seconds (default: 30.0)",
    "max_retries": "Maximum retry attempts on failure (default: 3)",
    "retry_backoff": "Exponential backoff multiplier for retries (default: 2.0)",
    "max_parallel_requests": "Maximum concurrent validation requests (default: 4)",
}


def discover_validator_types() -> dict[str, type]:
    """Dynamically discover all validator types and their param classes from data_designer.config.

    Returns:
        Dict mapping validator_type values to their params classes.
    """
    validator_type_enum = getattr(dd, "ValidatorType", None)
    if validator_type_enum is None or not issubclass(validator_type_enum, Enum):
        print(
            "Error: Could not find ValidatorType enum in data_designer.config",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build dict of all ValidatorParams classes by normalized name
    params_classes = {}
    for name in dir(dd):
        if name.endswith("ValidatorParams"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                # Normalize: CodeValidatorParams -> code
                normalized = name.replace("ValidatorParams", "").lower()
                params_classes[normalized] = obj

    # Map validator types to their params classes
    validator_types = {}
    for member in validator_type_enum:
        validator_name = member.name.lower()
        normalized_name = validator_name.replace("_", "")
        params_cls = params_classes.get(normalized_name)
        if params_cls is not None:
            validator_types[validator_name] = params_cls

    return validator_types


def main() -> None:
    """CLI entry point."""
    run_cli(
        discover_fn=discover_validator_types,
        type_key="validator_type",
        type_label="validator_type",
        class_label="params_class",
        default_descriptions=DEFAULT_FIELD_DESCRIPTIONS,
        script_name="get_validator_info.py",
        description="Print LLM-optimized context for data designer validator types.",
        header_title="Data Designer Validator Types Reference",
        examples=[
            "uv run get_validator_info.py all             # Print all validator types",
            "uv run get_validator_info.py code            # Print specific validator type",
            "uv run get_validator_info.py LOCAL_CALLABLE  # Case-insensitive lookup",
            "uv run get_validator_info.py --list          # List validator types and params classes",
        ],
        case_insensitive=True,
        uppercase_value=True,
    )


if __name__ == "__main__":
    main()
