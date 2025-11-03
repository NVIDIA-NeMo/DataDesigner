# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from .base import ConfigBase
from .datastore import DatastoreSettings
from .utils.io_helpers import validate_dataset_file_path


class SamplingStrategy(str, Enum):
    ORDERED = "ordered"
    SHUFFLE = "shuffle"


class IndexRange(ConfigBase):
    start: int = Field(..., ge=0)
    end: int = Field(..., ge=1)

    @model_validator(mode="after")
    def _validate_index_range(self) -> Self:
        if self.start >= self.end:
            raise ValueError("'start' index must be less than 'end' index")
        return self


class PartitionBlock(ConfigBase):
    partition_index: int = Field(..., default=0, ge=0)
    num_partitions: int = Field(..., default=1, ge=1)

    @model_validator(mode="after")
    def _validate_partition_block(self) -> Self:
        if self.partition_index >= self.num_partitions:
            raise ValueError("'partition_index' must be less than 'num_partitions'")
        return self


class SeedConfig(ConfigBase):
    dataset: str
    sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED
    selection_strategy: Optional[Union[IndexRange, PartitionBlock]] = None


class SeedDatasetReference(ABC, ConfigBase):
    dataset: str


class DatastoreSeedDatasetReference(SeedDatasetReference):
    datastore_settings: DatastoreSettings

    @property
    def repo_id(self) -> str:
        return "/".join(self.dataset.split("/")[:-1])

    @property
    def filename(self) -> str:
        return self.dataset.split("/")[-1]


class LocalSeedDatasetReference(SeedDatasetReference):
    @field_validator("dataset", mode="after")
    def validate_dataset_is_file(cls, v: str) -> str:
        return str(validate_dataset_file_path(v))
