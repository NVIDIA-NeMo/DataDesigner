"""Print LLM-optimized context for data designer sampler types.

Usage:
    uv run get_sampler_info.py              # Print help
    uv run get_sampler_info.py all          # Print all sampler types
    uv run get_sampler_info.py category     # Print specific sampler type
    uv run get_sampler_info.py UNIFORM      # Case-insensitive lookup
    uv run get_sampler_info.py --list       # List sampler types and params classes
"""

import inspect
import sys
from enum import Enum

from helpers.pydantic_info_utils import run_cli

import data_designer.config as dd

# Default descriptions for common sampler param fields that lack docstrings.
# These are derived from inspecting how fields are used in the codebase.
DEFAULT_FIELD_DESCRIPTIONS: dict[str, str] = {
    # Category sampler
    "values": "List of categorical values to sample from",
    "weights": "Optional sampling weights (probabilities) for each value",
    # Subcategory sampler
    "category": "Parent category column name for hierarchical sampling",
    # Person sampler
    "locale": "Locale for generating names (e.g., 'en_US')",
    "sex": "Filter by sex: 'Male' or 'Female'",
    "city": "Filter by city or list of cities",
    "age_range": "[min, max] age range for filtering",
    "with_synthetic_personas": "Include personality traits from Nemotron Personas",
    # UUID sampler
    "prefix": "Optional prefix string for generated UUIDs",
    "short_form": "Truncate UUID to 8 characters",
    "uppercase": "Use uppercase letters in UUID",
    # Datetime sampler
    "start": "Start date/datetime string (e.g., '2024-01-01')",
    "end": "End date/datetime string (e.g., '2024-12-31')",
    "unit": "Time unit: 'Y' (year), 'M' (month), 'D' (day), 'h', 'm', 's'",
    # TimeDelta sampler
    "reference_column_name": "Column name to compute time offset from",
    "dt_min": "Minimum time offset value",
    "dt_max": "Maximum time offset value",
    # Numeric samplers
    "low": "Minimum value (inclusive)",
    "high": "Maximum value (exclusive)",
    "mean": "Mean of the distribution",
    "stddev": "Standard deviation of the distribution",
    "decimal_places": "Number of decimal places for rounding output",
    "n": "Number of trials (for binomial distribution)",
    "p": "Probability of success (0.0 to 1.0)",
    # Scipy sampler
    "dist_name": "Name of scipy.stats distribution (e.g., 'expon', 'gamma')",
    "dist_params": "Dictionary of distribution parameters (e.g., {'scale': 5.0})",
}


def discover_sampler_types() -> dict[str, type]:
    """Dynamically discover all sampler types and their param classes from data_designer.config.

    Returns:
        Dict mapping sampler_type values (e.g., 'category') to their params classes.
    """
    # Find SamplerType enum
    sampler_type_enum = getattr(dd, "SamplerType", None)
    if sampler_type_enum is None or not issubclass(sampler_type_enum, Enum):
        print(
            "Error: Could not find SamplerType enum in data_designer.config",
            file=sys.stderr,
        )
        sys.exit(1)

    # Build dict of all SamplerParams classes by normalized name
    params_classes = {}
    for name in dir(dd):
        if name.endswith("SamplerParams"):
            obj = getattr(dd, name)
            if inspect.isclass(obj) and hasattr(obj, "model_fields"):
                # Normalize: CategorySamplerParams -> category
                normalized = name.replace("SamplerParams", "").lower()
                params_classes[normalized] = obj

    # Map sampler types to their params classes
    sampler_types = {}
    for member in sampler_type_enum:
        # member.name is like 'CATEGORY', member.value is the actual value
        sampler_name = member.name.lower()
        # Handle special cases like PERSON_FROM_FAKER -> personfromfaker
        normalized_name = sampler_name.replace("_", "")

        # Try to find matching params class
        params_cls = params_classes.get(normalized_name)
        if params_cls is None:
            # Try with underscores removed differently
            params_cls = params_classes.get(sampler_name.replace("_", ""))

        if params_cls is not None:
            sampler_types[sampler_name] = params_cls

    return sampler_types


def main() -> None:
    """CLI entry point."""
    run_cli(
        discover_fn=discover_sampler_types,
        type_key="sampler_type",
        type_label="sampler_type",
        class_label="params_class",
        default_descriptions=DEFAULT_FIELD_DESCRIPTIONS,
        script_name="get_sampler_info.py",
        description="Print LLM-optimized context for data designer sampler types.",
        header_title="Data Designer Sampler Types Reference",
        examples=[
            "uv run get_sampler_info.py all        # Print all sampler types",
            "uv run get_sampler_info.py category   # Print specific sampler type",
            "uv run get_sampler_info.py UNIFORM    # Case-insensitive lookup",
            "uv run get_sampler_info.py --list     # List sampler types and params classes",
        ],
        case_insensitive=True,
        uppercase_value=True,
    )


if __name__ == "__main__":
    main()
