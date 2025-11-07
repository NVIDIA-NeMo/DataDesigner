# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
import logging
import os
from pathlib import Path
from typing import Annotated, Literal, Optional, Union

from huggingface_hub import HfFileSystem
import pandas as pd
import pyarrow.parquet as pq
from pydantic import Field, field_validator, model_validator
from typing_extensions import Self, TypeAlias

from .base import ConfigBase
from .errors import InvalidFileFormatError, InvalidFilePathError
from .utils.io_helpers import (
    VALID_DATASET_FILE_EXTENSIONS,
    validate_dataset_file_path,
    validate_path_contains_files_of_type,
)

logger = logging.getLogger(__name__)


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
        source: Optional source name if you are running in a context with pre-registered, named
            sources from which seed datasets can be used.

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
    source: Optional[str] = None


class SeedDatasetReference(ABC, ConfigBase):
    @abstractmethod
    def get_dataset(self) -> str: ...

    @abstractmethod
    def get_source(self) -> Optional[str]: ...

    @abstractmethod
    def get_column_names(self) -> list[str]: ...


class LocalSeedDatasetReference(SeedDatasetReference):
    reference_type: Literal["local"] = "local"

    dataset: Union[str, Path]

    @field_validator("dataset", mode="after")
    def validate_dataset_is_file(cls, v: Union[str, Path]) -> Union[str, Path]:
        valid_wild_card_versions = {f"*{ext}" for ext in VALID_DATASET_FILE_EXTENSIONS}
        if any(str(v).endswith(wildcard) for wildcard in valid_wild_card_versions):
            parts = str(v).split("*.")
            file_path = parts[0]
            file_extension = parts[-1]
            validate_path_contains_files_of_type(file_path, file_extension)
        else:
            validate_dataset_file_path(v)
        return v

    def get_dataset(self) -> str:
        return str(self.dataset)

    def get_source(self) -> Optional[str]:
        return None

    def get_column_names(self) -> list[str]:
        file_type = Path(self.dataset).suffix.lower()[1:]
        return _get_file_column_names(self.dataset, file_type)


class HfHubSeedDatasetReference(SeedDatasetReference):
    reference_type: Literal["hf_hub"] = "hf_hub"

    dataset: str
    endpoint: str = "https://huggingface.co"
    token: Optional[str] = None
    source_name: Optional[str] = None

    def get_dataset(self) -> str:
        return self.dataset

    def get_source(self) -> Optional[str]:
        return self.source_name

    def get_column_names(self) -> list[str]:
        filename = self.dataset.split("/")[-1]
        repo_id = "/".join(self.dataset.split("/")[:-1])

        file_type = filename.split(".")[-1]
        if f".{file_type}" not in VALID_DATASET_FILE_EXTENSIONS:
            raise InvalidFileFormatError(f"🛑 Unsupported file type: {filename!r}")

        _token = self.token
        if self.token is not None:
            # Check if the value is an env var name and if so resolve it,
            # otherwise assume the value is the raw token string in plain text
            _token = os.environ.get(self.token, self.token)

        fs = HfFileSystem(endpoint=self.endpoint, token=_token)

        with fs.open(f"datasets/{repo_id}/{filename}") as f:
            return _get_file_column_names(f, file_type)


SeedDatasetReferenceT: TypeAlias = Annotated[
    Union[LocalSeedDatasetReference, HfHubSeedDatasetReference],
    Field(discriminator="reference_type"),
]


def _get_file_column_names(file_path: Union[str, Path], file_type: str) -> list[str]:
    """Extract column names based on file type."""
    file_path = Path(file_path)
    if "*" in str(file_path):
        matching_files = sorted(file_path.parent.glob(file_path.name))
        if not matching_files:
            raise InvalidFilePathError(f"🛑 No files found matching pattern: {str(file_path)!r}")
        logger.debug(f"0️⃣Using the first matching file in {str(file_path)!r} to determine column names in seed dataset")
        file_path = matching_files[0]

    if file_type == "parquet":
        try:
            schema = pq.read_schema(file_path)
            if hasattr(schema, "names"):
                return schema.names
            else:
                return [field.name for field in schema]
        except Exception as e:
            logger.warning(f"Failed to process parquet file {file_path}: {e}")
            return []
    elif file_type in ["json", "jsonl"]:
        return pd.read_json(file_path, orient="records", lines=True, nrows=1).columns.tolist()
    elif file_type == "csv":
        try:
            df = pd.read_csv(file_path, nrows=1)
            return df.columns.tolist()
        except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
            logger.warning(f"Failed to process CSV file {file_path}: {e}")
            return []
    else:
        raise InvalidFilePathError(f"🛑 Unsupported file type: {file_type!r}")
