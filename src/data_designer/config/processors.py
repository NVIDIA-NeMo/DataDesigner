# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import Literal

from pydantic import Field, field_validator

from data_designer.config.base import ConfigBase
from data_designer.config.dataset_builders import BuildStage

SUPPORTED_STAGES = [BuildStage.POST_BATCH]


class ProcessorType(str, Enum):
    DROP_COLUMNS = "drop_columns"
    OUTPUT_FORMAT = "output_format"


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
    elif processor_type == ProcessorType.OUTPUT_FORMAT:
        return OutputFormatProcessorConfig(**kwargs)


class DropColumnsProcessorConfig(ProcessorConfig):
    column_names: list[str]
    processor_type: Literal[ProcessorType.DROP_COLUMNS] = ProcessorType.DROP_COLUMNS


class OutputFormatProcessorConfig(ProcessorConfig):
    template: str = Field(
        ..., description="The Jinja template to use for each entry in the dataset, as a single string."
    )
    extension: str = Field(default="jsonl", description="The extension of the output files, e.g. 'jsonl' or 'csv'.")
    processor_type: Literal[ProcessorType.OUTPUT_FORMAT] = ProcessorType.OUTPUT_FORMAT
