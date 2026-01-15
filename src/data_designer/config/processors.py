# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from abc import ABC
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator
from typing_extensions import TypeAlias

from data_designer.config.base import ConfigBase
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.errors import InvalidConfigError

SUPPORTED_STAGES = [BuildStage.POST_BATCH]


class ProcessorType(str, Enum):
    """Enumeration of available processor types.

    Attributes:
        DROP_COLUMNS: Processor that removes specified columns from the output dataset.
        SCHEMA_TRANSFORM: Processor that creates a new dataset with a transformed schema using Jinja2 templates.
    """

    DROP_COLUMNS = "drop_columns"
    SCHEMA_TRANSFORM = "schema_transform"


class ProcessorConfig(ConfigBase, ABC):
    """Abstract base class for all processor configuration types.

    Processors are transformations that run before or after columns are generated.
    They can modify, reshape, or augment the dataset before it's saved.

    Attributes:
        name: Unique name of the processor, used to identify the processor in results
            and to name output artifacts on disk.
        build_stage: The stage at which the processor runs. Currently only `POST_BATCH`
            is supported, meaning processors run after each batch of columns is generated.
    """

    name: str = Field(
        description="The name of the processor, used to identify the processor in the results and to write the artifacts to disk.",
    )
    build_stage: BuildStage = Field(
        default=BuildStage.POST_BATCH,
        description=f"The stage at which the processor will run. Supported stages: {', '.join(SUPPORTED_STAGES)}",
    )
    processor_type: str

    @field_validator("build_stage")
    def validate_build_stage(cls, v: BuildStage) -> BuildStage:
        if v not in SUPPORTED_STAGES:
            raise ValueError(
                f"Invalid dataset builder stage: {v}. Only these stages are supported: {', '.join(SUPPORTED_STAGES)}"
            )
        return v


def get_processor_config_from_kwargs(processor_type: ProcessorType, **kwargs: Any) -> ProcessorConfig:
    """Create a processor configuration from a processor type and keyword arguments.

    Args:
        processor_type: The type of processor to create.
        **kwargs: Additional keyword arguments passed to the processor constructor.

    Returns:
        A processor configuration object of the specified type.
    """
    if processor_type == ProcessorType.DROP_COLUMNS:
        return DropColumnsProcessorConfig(**kwargs)
    elif processor_type == ProcessorType.SCHEMA_TRANSFORM:
        return SchemaTransformProcessorConfig(**kwargs)


class DropColumnsProcessorConfig(ProcessorConfig):
    """Configuration for dropping columns from the output dataset.

    This processor removes specified columns from the generated dataset. The dropped
    columns are saved separately in a `dropped-columns` directory for reference.
    When this processor is added via the config builder, the corresponding column
    configs are automatically marked with `drop = True`.

    Alternatively, you can set `drop = True` when configuring a column.

    Attributes:
        column_names: List of column names to remove from the output dataset.
        processor_type: Discriminator field, always `ProcessorType.DROP_COLUMNS` for this configuration type.
    """

    column_names: list[str] = Field(description="List of column names to drop from the output dataset.")
    processor_type: Literal[ProcessorType.DROP_COLUMNS] = ProcessorType.DROP_COLUMNS


class SchemaTransformProcessorConfig(ProcessorConfig):
    """Configuration for transforming the dataset schema using Jinja2 templates.

    This processor creates a new dataset with a transformed schema. Each key in the
    template becomes a column in the output, and values are Jinja2 templates that
    can reference any column in the batch. The transformed dataset is written to
    a `processors-outputs/{processor_name}/` directory alongside the main dataset.

    Attributes:
        template: Dictionary defining the output schema. Keys are new column names,
            values are Jinja2 templates (strings, lists, or nested structures).
            Must be JSON-serializable.
        processor_type: Discriminator field, always `ProcessorType.SCHEMA_TRANSFORM` for this configuration type.
    """

    template: dict[str, Any] = Field(
        ...,
        description="""
        Dictionary specifying columns and templates to use in the new dataset with transformed schema.

        Each key is a new column name, and each value is an object containing Jinja2 templates - for instance, a string or a list of strings.
        Values must be JSON-serializable.

        Example:

        ```python
        template = {
            "list_of_strings": ["{{ col1 }}", "{{ col2 }}"],
            "uppercase_string": "{{ col1 | upper }}",
            "lowercase_string": "{{ col2 | lower }}",
        }
        ```

        The above templates will create an new dataset with three columns: "list_of_strings", "uppercase_string", and "lowercase_string".
        References to columns "col1" and "col2" in the templates will be replaced with the actual values of the columns in the dataset.
        """,
    )
    processor_type: Literal[ProcessorType.SCHEMA_TRANSFORM] = ProcessorType.SCHEMA_TRANSFORM

    @field_validator("template")
    def validate_template(cls, v: dict[str, Any]) -> dict[str, Any]:
        try:
            json.dumps(v)
        except TypeError as e:
            if "not JSON serializable" in str(e):
                raise InvalidConfigError("Template must be JSON serializable")
        return v


ProcessorConfigT: TypeAlias = DropColumnsProcessorConfig | SchemaTransformProcessorConfig
