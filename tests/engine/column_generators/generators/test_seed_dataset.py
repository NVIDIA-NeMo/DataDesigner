# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import tempfile
from typing import TYPE_CHECKING
from unittest.mock import Mock, patch

import pytest

from data_designer.config.column_configs import SeedDatasetColumnConfig
from data_designer.config.seed import IndexRange, PartitionBlock, SamplingStrategy
from data_designer.config.seed_source import HuggingFaceSeedSource, LocalFileSeedSource
from data_designer.engine.column_generators.generators.base import GenerationStrategy
from data_designer.engine.column_generators.generators.seed_dataset import (
    MAX_ZERO_RECORD_RESPONSE_FACTOR,
    SeedDatasetColumnGenerator,
)
from data_designer.engine.column_generators.utils.errors import SeedDatasetError
from data_designer.engine.dataset_builders.multi_column_configs import SeedDatasetMultiColumnConfig
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import duckdb, pd

if TYPE_CHECKING:
    import duckdb
    import pandas as pd


@pytest.fixture
def stub_duckdb_conn():
    mock_conn = Mock()
    mock_conn.execute.return_value.fetchone.return_value = [1000]
    return mock_conn


@pytest.fixture
def stub_seed_dataset_config():
    return SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="col1")],
        source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
    )


@pytest.fixture
def stub_seed_dataset_generator(stub_resource_provider, stub_duckdb_conn, stub_seed_dataset_config):
    mock_provider = stub_resource_provider
    mock_seed_reader = mock_provider.seed_reader
    mock_seed_reader.create_duckdb_connection.return_value = stub_duckdb_conn
    mock_seed_reader.get_dataset_uri.return_value = "test_uri"

    return SeedDatasetColumnGenerator(config=stub_seed_dataset_config, resource_provider=mock_provider)


@pytest.fixture
def sample_dataframe():
    """Sample dataframe to be saved in different formats."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "name": ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"],
            "age": [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
            "city": [
                "New York",
                "Los Angeles",
                "Chicago",
                "Houston",
                "Phoenix",
                "Philadelphia",
                "San Antonio",
                "San Diego",
                "Dallas",
                "San Jose",
            ],
            "score": [85.5, 90.2, 78.9, 92.3, 88.7, 95.1, 82.4, 87.6, 91.8, 79.3],
        }
    )


@pytest.fixture
def seed_dataset_parquet(sample_dataframe):
    """Create a temporary parquet file."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False) as f:
        sample_dataframe.to_parquet(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def seed_dataset_csv(sample_dataframe):
    """Create a temporary CSV file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        sample_dataframe.to_csv(f.name, index=False)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def seed_dataset_json(sample_dataframe):
    """Create a temporary JSON file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        sample_dataframe.to_json(f.name, orient="records", lines=False, indent=2)
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def seed_dataset_jsonl(sample_dataframe):
    """Create a temporary JSONL file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        sample_dataframe.to_json(f.name, orient="records", lines=True)
        yield f.name
    os.unlink(f.name)


def test_seed_dataset_column_generator_generation_strategy() -> None:
    assert SeedDatasetColumnGenerator.get_generation_strategy() == GenerationStrategy.FULL_COLUMN


def test_seed_dataset_column_generator_config_structure():
    config = SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="col1"), SeedDatasetColumnConfig(name="col2")],
        source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )

    assert config.source.path == "hf://datasets/test/dataset"
    assert config.sampling_strategy == SamplingStrategy.SHUFFLE
    assert len(config.columns) == 2
    assert config.columns[0].name == "col1"
    assert config.columns[0].column_type == "seed-dataset"
    assert config.columns[1].name == "col2"
    assert config.columns[1].column_type == "seed-dataset"
    assert config.selection_strategy is None

    # Test PartitionBlock selection strategy
    config = SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="col1"), SeedDatasetColumnConfig(name="col2")],
        source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=PartitionBlock(index=1, num_partitions=3),
    )
    assert isinstance(config.selection_strategy, PartitionBlock)
    assert config.selection_strategy.index == 1
    assert config.selection_strategy.num_partitions == 3

    # Test IndexRange selection strategy
    config = SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="col1"), SeedDatasetColumnConfig(name="col2")],
        source=HuggingFaceSeedSource(path="hf://datasets/test/dataset"),
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=0, end=1),
    )
    assert isinstance(config.selection_strategy, IndexRange)
    assert config.selection_strategy.start == 0
    assert config.selection_strategy.end == 1
    assert config.selection_strategy.size == 2

    # Test constants and enum values
    assert MAX_ZERO_RECORD_RESPONSE_FACTOR == 2
    assert SamplingStrategy.ORDERED == "ordered"
    assert SamplingStrategy.SHUFFLE == "shuffle"


def test_seed_dataset_column_generator_generator_properties(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator
    mock_duckdb_conn = gen.duckdb_conn

    assert gen.num_records_sampled == 0
    gen._num_records_sampled = 100
    assert gen.num_records_sampled == 100

    conn1 = gen.duckdb_conn
    conn2 = gen.duckdb_conn
    assert conn1 == conn2 == mock_duckdb_conn


def test_seed_dataset_column_generator_generate_method(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator
    mock_generated_df = pd.DataFrame({"col1": [1, 2, 3]})
    gen.generate_from_scratch = Mock(return_value=mock_generated_df)

    input_df = pd.DataFrame({"existing": [4, 5, 6]})

    with patch("data_designer.engine.column_generators.generators.seed_dataset.concat_datasets") as mock_concat:
        mock_concat.return_value = pd.DataFrame({"col1": [1, 2, 3], "existing": [4, 5, 6]})

        gen.generate(input_df)

        gen.generate_from_scratch.assert_called_once_with(3)
        mock_concat.assert_called_once_with([mock_generated_df, input_df])


@pytest.mark.parametrize("num_records", [0, -1])
def test_seed_dataset_column_generator_generate_from_scratch_invalid_records(stub_seed_dataset_generator, num_records):
    gen = stub_seed_dataset_generator

    with pytest.raises(ValueError, match="🛑 `num_records` must be positive"):
        gen.generate_from_scratch(num_records)


def test_seed_dataset_column_generator_generate_from_scratch_valid_records(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    gen._reset_batch_reader = Mock()
    gen._sample_records = Mock(return_value=pd.DataFrame({"col1": [1, 2, 3]}))

    result = gen.generate_from_scratch(3)

    gen._reset_batch_reader.assert_called_once_with(3)
    gen._sample_records.assert_called_once_with(3)
    assert len(result) == 3


@pytest.mark.parametrize(
    "sampling_strategy,expected_shuffle",
    [
        (SamplingStrategy.SHUFFLE, True),
        (SamplingStrategy.ORDERED, False),
    ],
)
def test_seed_dataset_column_generator_reset_batch_reader(
    stub_seed_dataset_generator, stub_duckdb_conn, sampling_strategy, expected_shuffle
):
    gen = stub_seed_dataset_generator
    gen.config.sampling_strategy = sampling_strategy

    mock_query_result = Mock()
    mock_batch_reader = Mock()
    mock_query_result.record_batch.return_value = mock_batch_reader
    stub_duckdb_conn.query.return_value = mock_query_result

    gen._reset_batch_reader(100)

    # Verify query was called with correct SQL
    call_args = stub_duckdb_conn.query.call_args[0][0]
    if expected_shuffle:
        assert "ORDER BY RANDOM()" in call_args
    else:
        assert "ORDER BY RANDOM()" not in call_args

    assert "SELECT * FROM 'test_uri'" in call_args
    mock_query_result.record_batch.assert_called_once_with(batch_size=100)
    assert gen._batch_reader == mock_batch_reader


def test_seed_dataset_column_generator_sample_records_simple(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    # Mock batch reader to return data
    mock_batch = Mock()
    mock_batch.to_pandas.return_value = pd.DataFrame({"col1": [1, 2, 3]})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.return_value = mock_batch

    result = gen._sample_records(3)

    assert len(result) == 3
    assert list(result["col1"]) == [1, 2, 3]
    assert gen._num_records_sampled == 3
    assert gen._df_remaining is None


def test_seed_dataset_column_generator_sample_records_with_remaining(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    # Mock batch reader to return more data than requested
    mock_batch = Mock()
    mock_batch.to_pandas.return_value = pd.DataFrame({"col1": [1, 2, 3, 4, 5]})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.return_value = mock_batch

    result = gen._sample_records(3)

    assert len(result) == 3
    assert list(result["col1"]) == [1, 2, 3]
    assert gen._num_records_sampled == 3
    assert gen._df_remaining is not None
    assert len(gen._df_remaining) == 2
    assert list(gen._df_remaining["col1"]) == [4, 5]


def test_seed_dataset_column_generator_sample_records_with_previous_remaining(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator
    gen._df_remaining = pd.DataFrame({"col1": [10, 11]})

    # Mock batch reader to return additional data
    mock_batch = Mock()
    mock_batch.to_pandas.return_value = pd.DataFrame({"col1": [1, 2, 3]})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.return_value = mock_batch

    result = gen._sample_records(4)

    assert len(result) == 4
    assert list(result["col1"]) == [10, 11, 1, 2]
    assert gen._num_records_sampled == 4
    assert gen._df_remaining is not None
    assert len(gen._df_remaining) == 1
    assert list(gen._df_remaining["col1"]) == [3]


def test_seed_dataset_column_generator_sample_records_with_stop_iteration(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    # First call raises StopIteration, then returns data after reset
    mock_batch1 = Mock()
    mock_batch1.to_pandas.side_effect = StopIteration()

    mock_batch2 = Mock()
    mock_batch2.to_pandas.return_value = pd.DataFrame({"col1": [1, 2, 3]})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.side_effect = [mock_batch1, mock_batch2]

    gen._reset_batch_reader = Mock()

    result = gen._sample_records(3)

    # Verify reset was called when StopIteration occurred
    gen._reset_batch_reader.assert_called_once_with(3)
    assert len(result) == 3


def test_seed_dataset_column_generator_sample_records_zero_record_error(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    # Mock batch reader to always return empty dataframes
    mock_batch = Mock()
    mock_batch.to_pandas.return_value = pd.DataFrame({"col1": []})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.return_value = mock_batch

    with pytest.raises(RuntimeError, match="🛑 Something went wrong while reading from the datastore"):
        gen._sample_records(3)


def test_seed_dataset_column_generator_sample_records_multiple_batches(stub_seed_dataset_generator):
    gen = stub_seed_dataset_generator

    # Mock batch reader to return data in multiple batches
    mock_batch1 = Mock()
    mock_batch1.to_pandas.return_value = pd.DataFrame({"col1": [1, 2]})

    mock_batch2 = Mock()
    mock_batch2.to_pandas.return_value = pd.DataFrame({"col1": [3, 4]})

    gen._batch_reader = Mock()
    gen._batch_reader.read_next_batch.side_effect = [mock_batch1, mock_batch2]

    result = gen._sample_records(4)

    assert len(result) == 4
    assert list(result["col1"]) == [1, 2, 3, 4]
    assert gen._num_records_sampled == 4
    assert gen._df_remaining is None


# ============================================================================
# Tests for Different File Formats (parquet, csv, json, jsonl)
# ============================================================================


def create_generator_with_real_file(
    file_path: str,
    stub_resource_provider: ResourceProvider,
    sampling_strategy: SamplingStrategy = SamplingStrategy.ORDERED,
    selection_strategy: IndexRange | PartitionBlock | None = None,
) -> SeedDatasetColumnGenerator:
    """Helper function to create a generator with a real file and DuckDB connection."""
    config = SeedDatasetMultiColumnConfig(
        columns=[
            SeedDatasetColumnConfig(name="id"),
            SeedDatasetColumnConfig(name="name"),
            SeedDatasetColumnConfig(name="age"),
            SeedDatasetColumnConfig(name="city"),
            SeedDatasetColumnConfig(name="score"),
        ],
        source=LocalFileSeedSource(path=file_path),
        sampling_strategy=sampling_strategy,
        selection_strategy=selection_strategy,
    )

    # Create a real DuckDB connection (in-memory by default)
    real_conn = duckdb.connect()

    mock_provider = stub_resource_provider
    mock_seed_reader = mock_provider.seed_reader
    mock_seed_reader.create_duckdb_connection.return_value = real_conn
    mock_seed_reader.get_dataset_uri.return_value = file_path

    generator = SeedDatasetColumnGenerator(config=config, resource_provider=mock_provider)
    return generator


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_with_different_formats(fixture_name, stub_resource_provider, request):
    """Test that SeedDatasetColumnGenerator can read from different file formats."""
    # Get the fixture value using request.getfixturevalue
    file_path = request.getfixturevalue(fixture_name)

    generator = create_generator_with_real_file(file_path, stub_resource_provider)

    # Generate 5 records
    result = generator.generate_from_scratch(5)

    # Verify the results
    assert len(result) == 5
    assert list(result.columns) == ["id", "name", "age", "city", "score"]
    assert generator.num_records_sampled == 5

    # Verify data integrity by checking first row matches expected data
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[0]["age"] == 25
    assert result.iloc[0]["city"] == "New York"


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_ordered_sampling(fixture_name, stub_resource_provider, request):
    """Test ordered sampling strategy with different file formats."""
    file_path = request.getfixturevalue(fixture_name)

    config = SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="id"), SeedDatasetColumnConfig(name="name")],
        source=LocalFileSeedSource(path=file_path),
        sampling_strategy=SamplingStrategy.ORDERED,
    )

    real_conn = duckdb.connect()
    mock_provider = stub_resource_provider
    mock_provider.seed_reader.create_duckdb_connection.return_value = real_conn
    mock_provider.seed_reader.get_dataset_uri.return_value = file_path

    generator = SeedDatasetColumnGenerator(config=config, resource_provider=mock_provider)

    # Generate records twice to verify ordering is consistent
    result1 = generator.generate_from_scratch(3)
    result2 = generator.generate_from_scratch(3)

    assert len(result1) == 3
    assert len(result2) == 3
    assert list(result1["name"]) == ["Alice", "Bob", "Charlie"]
    assert list(result2["name"]) == ["David", "Eve", "Frank"]


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_shuffle_sampling(fixture_name, stub_resource_provider, request):
    """Test shuffle sampling strategy with different file formats."""
    file_path = request.getfixturevalue(fixture_name)

    config = SeedDatasetMultiColumnConfig(
        columns=[SeedDatasetColumnConfig(name="id"), SeedDatasetColumnConfig(name="name")],
        source=LocalFileSeedSource(path=file_path),
        sampling_strategy=SamplingStrategy.SHUFFLE,
    )

    real_conn = duckdb.connect()
    mock_provider = stub_resource_provider
    mock_provider.seed_reader.create_duckdb_connection.return_value = real_conn
    mock_provider.seed_reader.get_dataset_uri.return_value = file_path

    generator = SeedDatasetColumnGenerator(config=config, resource_provider=mock_provider)

    # Generate all records
    result = generator.generate_from_scratch(10)

    assert len(result) == 10
    # Verify all names are present (order may vary due to shuffle)
    assert set(result["name"]) == {"Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"}


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_cycling_through_dataset(fixture_name, stub_resource_provider, request):
    """Test that generator cycles through dataset when requesting more records than available."""
    file_path = request.getfixturevalue(fixture_name)

    generator = create_generator_with_real_file(file_path, stub_resource_provider)

    # Dataset has 10 records, request 15 to test cycling
    result = generator.generate_from_scratch(15)

    assert len(result) == 15
    assert generator.num_records_sampled == 15
    # First 10 should be the original data, next 5 should cycle back
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[10]["name"] == "Alice"  # Cycled back


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_partial_batch_with_remaining(fixture_name, stub_resource_provider, request):
    """Test that remaining records are properly handled across multiple generate calls."""
    file_path = request.getfixturevalue(fixture_name)

    generator = create_generator_with_real_file(file_path, stub_resource_provider)

    # Generate 7 records (less than the 10 available)
    result1 = generator.generate_from_scratch(7)
    assert len(result1) == 7
    assert list(result1["name"]) == ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace"]

    # Generate 5 more records (should use remaining 3 + cycle back for 2 more)
    result2 = generator.generate_from_scratch(5)
    assert len(result2) == 5
    assert list(result2["name"]) == ["Henry", "Ivy", "Jack", "Alice", "Bob"]

    assert generator.num_records_sampled == 12


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_generate_method_integration(fixture_name, stub_resource_provider, request):
    """Test the generate method that concatenates new data with existing dataset."""
    file_path = request.getfixturevalue(fixture_name)

    generator = create_generator_with_real_file(file_path, stub_resource_provider)

    # Create an existing dataset with different columns (concat_datasets concatenates horizontally)
    existing_df = pd.DataFrame({"existing_col1": ["val1", "val2"], "existing_col2": [100, 200]})

    result = generator.generate(existing_df)

    # Should have 2 rows (matching the existing dataset length)
    assert len(result) == 2
    # Should have all columns from both datasets
    assert set(result.columns) == {"id", "name", "age", "city", "score", "existing_col1", "existing_col2"}
    # Verify seed dataset columns are populated
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[1]["name"] == "Bob"
    # Verify existing columns are preserved
    assert result.iloc[0]["existing_col1"] == "val1"
    assert result.iloc[1]["existing_col2"] == 200


def test_seed_dataset_generator_dataset_size_detection_parquet(seed_dataset_parquet, stub_resource_provider):
    """Test that generator correctly detects dataset size for parquet files."""
    generator = create_generator_with_real_file(seed_dataset_parquet, stub_resource_provider)

    assert generator._seed_dataset_size == 10


def test_seed_dataset_generator_dataset_size_detection_csv(seed_dataset_csv, stub_resource_provider):
    """Test that generator correctly detects dataset size for CSV files."""
    generator = create_generator_with_real_file(seed_dataset_csv, stub_resource_provider)

    assert generator._seed_dataset_size == 10


def test_seed_dataset_generator_dataset_size_detection_json(seed_dataset_json, stub_resource_provider):
    """Test that generator correctly detects dataset size for JSON files."""
    generator = create_generator_with_real_file(seed_dataset_json, stub_resource_provider)

    assert generator._seed_dataset_size == 10


def test_seed_dataset_generator_dataset_size_detection_jsonl(seed_dataset_jsonl, stub_resource_provider):
    """Test that generator correctly detects dataset size for JSONL files."""
    generator = create_generator_with_real_file(seed_dataset_jsonl, stub_resource_provider)

    assert generator._seed_dataset_size == 10


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_uses_real_duckdb_connection(fixture_name, stub_resource_provider, request):
    """Test that generator uses a real DuckDB connection to read actual file data."""
    file_path = request.getfixturevalue(fixture_name)

    generator = create_generator_with_real_file(file_path, stub_resource_provider)

    # Verify the duckdb_conn is a real connection
    assert isinstance(generator.duckdb_conn, duckdb.DuckDBPyConnection)

    # Verify we can query the file directly through the connection
    result = generator.duckdb_conn.execute(f"SELECT * FROM '{file_path}' LIMIT 3").fetchdf()

    assert len(result) == 3
    assert "name" in result.columns
    assert result.iloc[0]["name"] == "Alice"
    assert result.iloc[1]["name"] == "Bob"
    assert result.iloc[2]["name"] == "Charlie"

    # Verify the dataset URI is set correctly
    assert generator._dataset_uri == file_path

    # Verify the connection can execute count queries
    count_result = generator.duckdb_conn.execute(f"SELECT COUNT(*) FROM '{file_path}'").fetchone()[0]
    assert count_result == 10


# ============================================================================
# Tests for SeedConfig selection strategies
# ============================================================================
@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_index_range_selection_strategy(fixture_name, stub_resource_provider, request):
    """Test that generator correctly applies index range selection strategy."""
    # Ordered Sampling

    # Range with a subset of items
    file_path = request.getfixturevalue(fixture_name)
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=IndexRange(start=4, end=8),
    )
    result = generator.generate_from_scratch(6)
    assert len(result) == 6
    assert list(result["name"]) == ["Eve", "Frank", "Grace", "Henry", "Ivy", "Eve"]

    # Range with just one item
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=IndexRange(start=4, end=4),
    )
    result = generator.generate_from_scratch(1)
    assert len(result) == 1
    assert list(result["name"]) == ["Eve"]

    # Range with all items
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=IndexRange(start=0, end=9),
    )
    result = generator.generate_from_scratch(10)
    assert len(result) == 10
    assert list(result["name"]) == ["Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"]

    # Shuffle Sampling

    # Range with a subset of items
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=4, end=8),
    )
    result = generator.generate_from_scratch(10)
    assert len(result) == 10
    assert set(result["name"]).issubset({"Eve", "Frank", "Grace", "Henry", "Ivy"})

    # Range with just one item
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=4, end=4),
    )
    result = generator.generate_from_scratch(1)
    assert len(result) == 1
    assert list(result["name"]) == ["Eve"]

    # Range with all items
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=IndexRange(start=0, end=9),
    )
    result = generator.generate_from_scratch(10)
    assert len(result) == 10
    assert set(result["name"]).issubset(
        {"Alice", "Bob", "Charlie", "David", "Eve", "Frank", "Grace", "Henry", "Ivy", "Jack"}
    )


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_partition_block_selection_strategy(fixture_name, stub_resource_provider, request):
    """Test that generator correctly applies partition block selection strategy."""
    file_path = request.getfixturevalue(fixture_name)
    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.ORDERED,
        selection_strategy=PartitionBlock(index=1, num_partitions=3),
    )
    result = generator.generate_from_scratch(5)
    assert len(result) == 5
    # Requesting 5 items from a 3-item partition should cycle:
    assert list(result["name"]) == ["David", "Eve", "Frank", "David", "Eve"]

    generator = create_generator_with_real_file(
        file_path,
        stub_resource_provider,
        sampling_strategy=SamplingStrategy.SHUFFLE,
        selection_strategy=PartitionBlock(index=4, num_partitions=5),
    )
    result = generator.generate_from_scratch(10)
    assert len(result) == 10
    assert set(result["name"]).issubset({"Jack", "Ivy"})


@pytest.mark.parametrize(
    "fixture_name",
    [
        "seed_dataset_parquet",
        "seed_dataset_csv",
        "seed_dataset_json",
        "seed_dataset_jsonl",
    ],
)
def test_seed_dataset_generator_invalid_selection_strategies(fixture_name, stub_resource_provider, request):
    """Test that generator raises an error for invalid selection strategies."""
    file_path = request.getfixturevalue(fixture_name)
    with pytest.raises(
        SeedDatasetError, match="Selection strategy 'end' index 10 is out of bounds for dataset size 10"
    ):
        generator = create_generator_with_real_file(
            file_path, stub_resource_provider, selection_strategy=IndexRange(start=1, end=10)
        )
        generator.generate_from_scratch(1)
    with pytest.raises(
        SeedDatasetError, match="Selection strategy 'num_partitions' 11 is out of bounds for dataset size 10"
    ):
        generator = create_generator_with_real_file(
            file_path, stub_resource_provider, selection_strategy=PartitionBlock(index=0, num_partitions=11)
        )
        generator.generate_from_scratch(1)
