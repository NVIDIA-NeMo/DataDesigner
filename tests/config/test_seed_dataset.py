# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import pandas as pd
import pytest

from data_designer.config.errors import InvalidFilePathError
from data_designer.config.seed_dataset import LocalFileSeedConfig


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
        LocalFileSeedConfig(path="test/dataset.parquet")

    # Should not raise an error when referencing supported extensions with wildcard pattern.
    create_partitions_in_path(tmp_path, "parquet")
    create_partitions_in_path(tmp_path, "csv")
    create_partitions_in_path(tmp_path, "json")
    create_partitions_in_path(tmp_path, "jsonl")

    test_cases = ["parquet", "csv", "json", "jsonl"]
    try:
        for extension in test_cases:
            config = LocalFileSeedConfig(path=f"{tmp_path}/*.{extension}")
            assert config.path == f"{tmp_path}/*.{extension}"
    except Exception as e:
        pytest.fail(f"Expected no exception, but got {e}")


def test_local_seed_dataset_reference_validation_error(tmp_path: Path):
    create_partitions_in_path(tmp_path, "parquet")
    with pytest.raises(InvalidFilePathError, match="does not contain files of type 'csv'"):
        LocalFileSeedConfig(path=f"{tmp_path}/*.csv")
