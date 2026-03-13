# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from data_designer.engine.resources.nemotron_personas_reader import (
    DATASETS_ROOT,
    LocalNemotronPersonasDatasetReader,
    init_nemotron_personas_reader,
)


@pytest.fixture
def stub_reader(stub_temp_dir: Path) -> LocalNemotronPersonasDatasetReader:
    return LocalNemotronPersonasDatasetReader(stub_temp_dir)


def test_local_reader_get_dataset_uri(stub_reader: LocalNemotronPersonasDatasetReader, stub_temp_dir: Path) -> None:
    uri = stub_reader.get_dataset_uri("en_US")
    assert uri == f"{stub_temp_dir}/{DATASETS_ROOT}/en_US.parquet"


def test_local_reader_get_dataset_uri_different_locale(
    stub_reader: LocalNemotronPersonasDatasetReader, stub_temp_dir: Path
) -> None:
    uri = stub_reader.get_dataset_uri("ja_JP")
    assert uri == f"{stub_temp_dir}/{DATASETS_ROOT}/ja_JP.parquet"


@patch("data_designer.engine.resources.nemotron_personas_reader.lazy.duckdb", autospec=True)
def test_local_reader_create_duckdb_connection(
    mock_duckdb: object, stub_reader: LocalNemotronPersonasDatasetReader
) -> None:
    stub_reader.create_duckdb_connection()
    mock_duckdb.connect.assert_called_once_with(config={"threads": 1, "memory_limit": "2 gb"})


def test_init_nemotron_personas_reader_returns_local_reader(stub_temp_dir: Path) -> None:
    reader = init_nemotron_personas_reader(str(stub_temp_dir))
    assert isinstance(reader, LocalNemotronPersonasDatasetReader)


def test_init_nemotron_personas_reader_nonexistent_path() -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        init_nemotron_personas_reader("/nonexistent/path")
