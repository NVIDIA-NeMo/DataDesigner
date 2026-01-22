# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import MagicMock

from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.resources.seed_reader import SeedReader


def test_dataset_metadata_defaults() -> None:
    """Test DatasetMetadata default values."""
    metadata = DatasetMetadata()
    assert metadata.seed_column_names == []


def test_dataset_metadata_with_seed_columns() -> None:
    """Test DatasetMetadata with seed column names."""
    metadata = DatasetMetadata(seed_column_names=["name", "age", "city"])
    assert metadata.seed_column_names == ["name", "age", "city"]


def test_get_dataset_metadata_with_seed_reader() -> None:
    """Test creating DatasetMetadata from ResourceProvider with seed reader."""
    seed_reader = MagicMock(spec=SeedReader)
    seed_reader.get_column_names.return_value = ["col1", "col2"]

    resource_provider = MagicMock(spec=ResourceProvider)
    resource_provider.seed_reader = seed_reader

    metadata = ResourceProvider.get_dataset_metadata(resource_provider)

    assert metadata.seed_column_names == ["col1", "col2"]
    seed_reader.get_column_names.assert_called_once()


def test_get_dataset_metadata_without_seed_reader() -> None:
    """Test creating DatasetMetadata from ResourceProvider without seed reader."""
    resource_provider = MagicMock(spec=ResourceProvider)
    resource_provider.seed_reader = None

    metadata = ResourceProvider.get_dataset_metadata(resource_provider)

    assert metadata.seed_column_names == []


def test_dataset_metadata_is_serializable() -> None:
    """Test that DatasetMetadata can be serialized to JSON."""
    metadata = DatasetMetadata(seed_column_names=["name", "age"])

    json_data = metadata.model_dump_json()
    assert '"seed_column_names":["name","age"]' in json_data

    # Can be deserialized back
    restored = DatasetMetadata.model_validate_json(json_data)
    assert restored.seed_column_names == ["name", "age"]
