# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC
from enum import Enum
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator
from typing_extensions import Self

from .base import ConfigBase
from .datastore import DatastoreSettings
from .utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)


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
    index: int = Field(default=0, ge=0, description="The index of the partition to sample from")
    num_partitions: int = Field(default=1, ge=1, description="The total number of partitions in the dataset")

    @model_validator(mode="after")
    def _validate_partition_block(self) -> Self:
        if self.index >= self.num_partitions:
            raise ValueError("'index' must be less than 'num_partitions'")
        return self

    def to_index_range(self, dataset_size: int) -> IndexRange:
        partition_size = dataset_size // self.num_partitions
        start = self.index * partition_size

        # For the last partition, extend to the end of the dataset to include remainder rows
        if self.index == self.num_partitions - 1:
            end = dataset_size - 1
        else:
            end = ((self.index + 1) * partition_size) - 1
        return IndexRange(start=start, end=end)


class SeedConfig(ConfigBase):
    """Configuration for sampling data from a seed dataset.

    Args:
        dataset: Path or identifier for the seed dataset.
        sampling_strategy: Strategy for how to sample rows from the dataset.
            - ORDERED: Read rows sequentially in their original order.
            - SHUFFLE: Randomly shuffle rows before sampling. When used with
              selection_strategy, shuffling occurs within the selected range/partition.
        selection_strategy: Optional strategy to select a subset of the dataset.
            - IndexRange: Select a specific range of indices (e.g., rows 100-200).
            - PartitionBlock: Select a partition by splitting the dataset into N equal parts.
              Partition indices are zero-based (index=0 is the first partition, index=1 is
              the second, etc.).

    Examples:
        Read rows sequentially from start to end:
            SeedConfig(dataset="my_data.parquet", sampling_strategy=SamplingStrategy.ORDERED)

        Read rows in random order:
            SeedConfig(dataset="my_data.parquet", sampling_strategy=SamplingStrategy.SHUFFLE)

        Read specific index range (rows 100-199):
            SeedConfig(
                dataset="my_data.parquet",
                sampling_strategy=SamplingStrategy.ORDERED,
                selection_strategy=IndexRange(start=100, end=199)
            )

        Read random rows from a specific index range (shuffles within rows 100-199):
            SeedConfig(
                dataset="my_data.parquet",
                sampling_strategy=SamplingStrategy.SHUFFLE,
                selection_strategy=IndexRange(start=100, end=199)
            )

        Read from partition 2 (3rd partition, zero-based) of 5 partitions (20% of dataset):
            SeedConfig(
                dataset="my_data.parquet",
                sampling_strategy=SamplingStrategy.ORDERED,
                selection_strategy=PartitionBlock(index=2, num_partitions=5)
            )

        Read shuffled rows from partition 0 of 10 partitions (shuffles within the partition):
            SeedConfig(
                dataset="my_data.parquet",
                sampling_strategy=SamplingStrategy.SHUFFLE,
                selection_strategy=PartitionBlock(index=0, num_partitions=10)
            )
    """

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
        valid_wild_card_versions = {f"*{ext}" for ext in VALID_DATASET_FILE_EXTENSIONS}
        if any(v.endswith(wildcard) for wildcard in valid_wild_card_versions):
            parts = v.split("*.")
            file_path = parts[0]
            file_extension = parts[-1]
            validate_path_contains_files_of_type(file_path, file_extension)
        else:
            validate_dataset_file_path(v)
        return v
