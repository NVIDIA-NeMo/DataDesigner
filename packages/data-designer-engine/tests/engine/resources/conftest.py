# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.engine.resources.person_reader import LocalPersonReader


@pytest.fixture
def stub_temp_dir() -> Path:
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def stub_person_reader(stub_temp_dir: Path) -> LocalPersonReader:
    return LocalPersonReader(stub_temp_dir)


@pytest.fixture
def stub_sample_dataframe() -> lazy.pd.DataFrame:
    return lazy.pd.DataFrame({"id": [1, 2, 3], "name": ["Alice", "Bob", "Charlie"], "age": [25, 30, 35]})


@pytest.fixture
def stub_artifact_storage() -> Mock:
    mock_storage = Mock()
    mock_storage.write_parquet_file = Mock()
    return mock_storage


@pytest.fixture
def stub_model_registry() -> Mock:
    mock_registry = Mock()
    mock_registry.get_model = Mock()
    mock_registry.get_model_config = Mock()
    return mock_registry


@pytest.fixture
def stub_secret_resolver() -> Mock:
    mock_resolver = Mock()
    mock_resolver.resolve = Mock(return_value="resolved_secret")
    return mock_resolver


@pytest.fixture
def write_jsonl() -> Callable[[Path, list[dict[str, Any]]], None]:
    def _write(path: Path, records: list[dict[str, Any]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for record in records:
                file.write(f"{json.dumps(record)}\n")

    return _write
