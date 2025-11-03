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
    start: int = Field(ge=0, description="The start index of the index range (inclusive)")
    end: int = Field(ge=0, description="The end index of the index range (inclusive)")

    @model_validator(mode="after")
    def _validate_index_range(self) -> Self:
        if self.start > self.end:
            raise ValueError("'start' index must be less than or equal to 'end' index")
        return self

    @property
    def size(self) -> int:
        return self.end - self.start + 1


class PartitionBlock(ConfigBase):
    partition_index: int = Field(default=0, ge=0, description="The index of the partition to sample from")
    num_partitions: int = Field(default=1, ge=1, description="The total number of partitions in the dataset")

    @model_validator(mode="after")
    def _validate_partition_block(self) -> Self:
        if self.partition_index >= self.num_partitions:
            raise ValueError("'partition_index' must be less than 'num_partitions'")
        return self

    def to_index_range(self, dataset_size: int) -> IndexRange:
        partition_size = dataset_size // self.num_partitions
        start = self.partition_index * partition_size

        # For the last partition, extend to the end of the dataset to include remainder rows
        if self.partition_index == self.num_partitions - 1:
            end = dataset_size - 1
        else:
            end = ((self.partition_index + 1) * partition_size) - 1
        return IndexRange(start=start, end=end)


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
