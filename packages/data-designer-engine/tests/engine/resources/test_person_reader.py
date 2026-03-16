# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_designer.engine.resources.person_reader import (
    DATASETS_ROOT,
    MEMORY_LIMIT,
    THREADS,
    LocalPersonReader,
    create_person_reader,
)


@pytest.fixture
def stub_reader(stub_temp_dir: Path) -> LocalPersonReader:
    return LocalPersonReader(stub_temp_dir)


def test_local_reader_get_dataset_uri(stub_reader: LocalPersonReader, stub_temp_dir: Path) -> None:
    uri = stub_reader.get_dataset_uri("en_US")
    assert uri == f"{stub_temp_dir}/{DATASETS_ROOT}/en_US.parquet"


@patch("data_designer.engine.resources.person_reader.lazy.duckdb", autospec=True)
def test_local_reader_create_duckdb_connection(
    mock_duckdb: object, stub_reader: LocalPersonReader
) -> None:
    stub_reader.create_duckdb_connection()
    mock_duckdb.connect.assert_called_once_with(config={"threads": THREADS, "memory_limit": MEMORY_LIMIT})


def test_create_person_reader_returns_local_reader(stub_temp_dir: Path) -> None:
    reader = create_person_reader(str(stub_temp_dir))
    assert isinstance(reader, LocalPersonReader)


def test_create_person_reader_nonexistent_path() -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        create_person_reader("/nonexistent/path")
