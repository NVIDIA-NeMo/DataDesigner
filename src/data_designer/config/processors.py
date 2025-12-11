# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from abc import ABC
from enum import Enum
from typing import Any, Literal

from pydantic import Field, field_validator

from data_designer.config.base import ConfigBase
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.errors import InvalidConfigError

SUPPORTED_STAGES = [BuildStage.POST_BATCH]


class ProcessorType(str, Enum):
    DROP_COLUMNS = "drop_columns"
    SCHEMA_TRANSFORM = "schema_transform"


class ProcessorConfig(ConfigBase, ABC):
    name: str = Field(
        description="The name of the processor, used to identify the processor in the results and to write the artifacts to disk.",
    )
    build_stage: BuildStage = Field(
        default=BuildStage.POST_BATCH,
        description=f"The stage at which the processor will run. Supported stages: {', '.join(SUPPORTED_STAGES)}",
    )

    @field_validator("build_stage")
    def validate_build_stage(cls, v: BuildStage) -> BuildStage:
        if v not in SUPPORTED_STAGES:
            raise ValueError(
                f"Invalid dataset builder stage: {v}. Only these stages are supported: {', '.join(SUPPORTED_STAGES)}"
            )
        return v


def get_processor_config_from_kwargs(processor_type: ProcessorType, **kwargs) -> ProcessorConfig:
    if processor_type == ProcessorType.DROP_COLUMNS:
        return DropColumnsProcessorConfig(**kwargs)
    elif processor_type == ProcessorType.SCHEMA_TRANSFORM:
        return SchemaTransformProcessorConfig(**kwargs)


class DropColumnsProcessorConfig(ProcessorConfig):
    column_names: list[str]
    processor_type: Literal[ProcessorType.DROP_COLUMNS] = ProcessorType.DROP_COLUMNS


class SchemaTransformProcessorConfig(ProcessorConfig):
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
