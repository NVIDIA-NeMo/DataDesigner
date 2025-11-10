# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed import IndexRange, LocalSeedDatasetReference, PartitionBlock


def create_partitions_in_path(temp_dir: Path, extension: str, num_files: int = 2) -> Path:
    df = pd.DataFrame({"col": [1, 2, 3]})

    for i in range(num_files):
        file_path = temp_dir / f"partition_{i}.{extension}"
        if extension == "parquet":
            df.to_parquet(file_path)
        elif extension == "csv":
            df.to_csv(file_path, index=False)
        elif extension == "json":
            df.to_json(file_path, orient="records", lines=True)
        elif extension == "jsonl":
            df.to_json(file_path, orient="records", lines=True)
    return temp_dir


def test_index_range_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=-1, end=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        IndexRange(start=0, end=-1)

    with pytest.raises(ValueError, match="'start' index must be less than or equal to 'end' index"):
        IndexRange(start=11, end=10)


def test_index_range_size():
    assert IndexRange(start=0, end=10).size == 11
    assert IndexRange(start=1, end=10).size == 10
    assert IndexRange(start=0, end=0).size == 1


def test_partition_block_validation():
    with pytest.raises(ValueError, match="should be greater than or equal to 0"):
        PartitionBlock(index=-1, num_partitions=10)

    with pytest.raises(ValueError, match="should be greater than or equal to 1"):
        PartitionBlock(index=0, num_partitions=0)

    with pytest.raises(ValueError, match="'index' must be less than 'num_partitions'"):
        PartitionBlock(index=10, num_partitions=10)


def test_partition_block_to_index_range():
    index_range = PartitionBlock(index=0, num_partitions=10).to_index_range(101)
    assert index_range.start == 0
    assert index_range.end == 9
    assert index_range.size == 10

    index_range = PartitionBlock(index=1, num_partitions=10).to_index_range(105)
    assert index_range.start == 10
    assert index_range.end == 19
    assert index_range.size == 10

    index_range = PartitionBlock(index=2, num_partitions=10).to_index_range(105)
    assert index_range.start == 20
    assert index_range.end == 29
    assert index_range.size == 10

    index_range = PartitionBlock(index=9, num_partitions=10).to_index_range(105)
    assert index_range.start == 90
    assert index_range.end == 104
    assert index_range.size == 15


def test_local_seed_dataset_reference_validation(tmp_path: Path):
    with pytest.raises(InvalidFilePathError, match="🛑 Path test/dataset.parquet is not a file."):
        LocalSeedDatasetReference(dataset="test/dataset.parquet")

    # Should not raise an error when referencing supported extensions with wildcard pattern.
    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    test_cases = ["parquet", "csv", "json", "jsonl"]
    try:
        for extension in test_cases:
            reference = LocalSeedDatasetReference(dataset=f"{tmp_path}/*.{extension}")
            assert reference.dataset == f"{tmp_path}/*.{extension}"
    except Exception as e:
        pytest.fail(f"Expected no exception, but got {e}")


def test_local_seed_dataset_reference_validation_error(tmp_path: Path):
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalSeedDatasetReference(dataset=f"{tmp_path}/*.csv")
