# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed_source import DataFrameSeedSource, LocalFileSeedSource
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


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


def test_local_seed_dataset_reference_validation(tmp_path: Path):
    with pytest.raises(InvalidFilePathError, match="🛑 Path test/dataset.parquet is not a file."):
        LocalFileSeedSource(path="test/dataset.parquet")

    # Should not raise an error when referencing supported extensions with wildcard pattern.
    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    test_cases = ["parquet", "csv", "json", "jsonl"]
    try:
        for extension in test_cases:
            config = LocalFileSeedSource(path=f"{tmp_path}/*.{extension}")
            assert config.path == f"{tmp_path}/*.{extension}"
    except Exception as e:
        pytest.fail(f"Expected no exception, but got {e}")


def test_local_seed_dataset_reference_validation_error(tmp_path: Path):
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalFileSeedSource(path=f"{tmp_path}/*.csv")


def test_local_source_from_dataframe(tmp_path: Path):
    df = pd.DataFrame({"col": [1, 2, 3]})
    filepath = f"{tmp_path}/data.parquet"

    source = LocalFileSeedSource.from_dataframe(df, filepath)

    assert source.path == filepath
    pd.testing.assert_frame_equal(df, pd.read_parquet(filepath))


def test_dataframe_seed_source_serialization():
    """Test that DataFrameSeedSource excludes the DataFrame field during serialization."""
    df = pd.DataFrame({"col1": [1, 2, 3], "col2": ["a", "b", "c"]})
    source = DataFrameSeedSource(df=df)

    # Test model_dump excludes the df field
    serialized = source.model_dump(mode="json")
    assert "df" not in serialized
    assert serialized == {"seed_type": "df"}
