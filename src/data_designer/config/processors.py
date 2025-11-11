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
    TO_JSONL = "to_jsonl"


class ProcessorConfig(ConfigBase, ABC):
    build_stage: BuildStage = Field(
        default=BuildStage.POST_BATCH,
        description=f"The stage at which the processor will run. Supported stages: {', '.join(SUPPORTED_STAGES)}"
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
    elif processor_type == ProcessorType.TO_JSONL:
        return ToJsonlProcessorConfig(**kwargs)


class DropColumnsProcessorConfig(ProcessorConfig):
    column_names: list[str]
    processor_type: Literal[ProcessorType.DROP_COLUMNS] = ProcessorType.DROP_COLUMNS


class ToJsonlProcessorConfig(ProcessorConfig):
    template: dict = Field(..., description="The template to use for each entry in the dataset.")
    folder_name: str = Field(..., description="Folder where JSONL files will be saved.")
    fraction_per_file: dict[str, float] = Field(
        default={"train.jsonl": 0.8, "validation.jsonl": 0.2},
        description="Fraction of the dataset to save in each file. The keys are the filenames and the values are the fractions.",
    )
    processor_type: Literal[ProcessorType.TO_JSONL] = ProcessorType.TO_JSONL

    @field_validator("fraction_per_file")
    def validate_fraction_per_file(cls, v: dict[str, float]) -> dict[str, float]:
        if sum(v.values()) != 1:
            raise ValueError("The fractions must sum to 1.")
        return v