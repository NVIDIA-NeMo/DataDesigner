"""Print LLM-optimized context for data designer processor types.

Usage:
    uv run get_processor_info.py              # Print help
    uv run get_processor_info.py all          # Print all processor types
    uv run get_processor_info.py drop_columns # Print specific processor type
    uv run get_processor_info.py --list       # List processor types and config classes
"""

import inspect
from enum import Enum
from typing import Literal, get_args, get_origin

from helpers.pydantic_info_utils import run_cli

import data_designer.config as dd

DEFAULT_FIELD_DESCRIPTIONS: dict[str, str] = {
    "name": "Unique processor name, used for artifact paths on disk",
    "build_stage": "Processing stage: currently only POST_BATCH is supported",
    "processor_type": "Discriminator identifying the processor type",
    "column_names": "List of column names to drop from the output dataset",
    "template": "Dict mapping new column names to Jinja2 template values (supports nested JSON)",
}


def discover_processor_configs() -> dict[str, type]:
    """Dynamically discover all ProcessorConfig classes from data_designer.config.

    Returns:
        Dict mapping processor_type values to their config classes.
    """
    processor_configs = {}
    for name in dir(dd):
        if name.endswith("ProcessorConfig") and name != "ProcessorConfig":
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                if "processor_type" in obj.model_fields:
                    annotation = obj.model_fields["processor_type"].annotation
                    if get_origin(annotation) is Literal:
                        args = get_args(annotation)
                        if args:
                            key = args[0].value if isinstance(args[0], Enum) else args[0]
                            processor_configs[key] = obj
    return processor_configs


def main() -> None:
    """CLI entry point."""
    run_cli(
        discover_fn=discover_processor_configs,
        type_key="processor_type",
        type_label="processor_type",
        class_label="config_class",
        default_descriptions=DEFAULT_FIELD_DESCRIPTIONS,
        script_name="get_processor_info.py",
        description="Print LLM-optimized context for data designer processor types.",
        header_title="Data Designer Processor Types Reference",
        examples=[
            "uv run get_processor_info.py all            # Print all processor types",
            "uv run get_processor_info.py drop_columns   # Print specific processor type",
            "uv run get_processor_info.py --list         # List processor types and config classes",
        ],
        case_insensitive=True,
        uppercase_value=False,
    )


if __name__ == "__main__":
    main()
