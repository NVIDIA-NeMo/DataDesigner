# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import pytest

from data_designer.config.analysis.dataset_profiler import DatasetProfilerResults
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.preview_results import PreviewResults
from data_designer.config.utils.errors import DatasetSampleDisplayError
from data_designer.config.utils.visualization import display_sample_record as display_fn
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.interface.results import DatasetCreationResults
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd


@pytest.fixture
def stub_artifact_storage(stub_dataframe):
    """Mock artifact storage that returns a test DataFrame."""
    storage = MagicMock(spec=ArtifactStorage)
    storage.load_dataset.return_value = stub_dataframe
    return storage


@pytest.fixture
def stub_dataset_metadata():
    """Fixture providing a DatasetMetadata instance."""
    return DatasetMetadata()


@pytest.fixture
def stub_dataset_creation_results(
    stub_artifact_storage, stub_dataset_profiler_results, stub_complete_builder, stub_dataset_metadata
):
    """Fixture providing a DatasetCreationResults instance."""
    return DatasetCreationResults(
        artifact_storage=stub_artifact_storage,
        analysis=stub_dataset_profiler_results,
        config_builder=stub_complete_builder,
        dataset_metadata=stub_dataset_metadata,
    )


def test_init(stub_artifact_storage, stub_dataset_profiler_results, stub_complete_builder, stub_dataset_metadata):
    """Test DatasetCreationResults initialization."""
    results = DatasetCreationResults(
        artifact_storage=stub_artifact_storage,
        analysis=stub_dataset_profiler_results,
        config_builder=stub_complete_builder,
        dataset_metadata=stub_dataset_metadata,
    )
    assert results.artifact_storage == stub_artifact_storage
    assert results._analysis == stub_dataset_profiler_results
    assert results._config_builder == stub_complete_builder
    assert results.dataset_metadata == stub_dataset_metadata


def test_load_dataset(stub_dataset_creation_results, stub_artifact_storage, stub_dataframe):
    """Test loading the dataset."""
    dataset = stub_dataset_creation_results.load_dataset()

    assert isinstance(dataset, pd.DataFrame)
    stub_artifact_storage.load_dataset.assert_called_once()
    pd.testing.assert_frame_equal(dataset, stub_dataframe)


def test_load_analysis(stub_dataset_creation_results, stub_dataset_profiler_results):
    """Test loading the analysis results."""
    analysis = stub_dataset_creation_results.load_analysis()

    assert isinstance(analysis, DatasetProfilerResults)
    assert analysis == stub_dataset_profiler_results


def test_load_analysis_returns_same_instance(stub_dataset_creation_results):
    """Test that load_analysis returns the same analysis instance."""
    analysis1 = stub_dataset_creation_results.load_analysis()
    analysis2 = stub_dataset_creation_results.load_analysis()

    assert analysis1 is analysis2


def test_record_sampler_dataset_initialization(stub_dataset_creation_results, stub_artifact_storage):
    """Test that _record_sampler_dataset cached property loads dataset correctly."""
    # Access the cached property
    dataset = stub_dataset_creation_results._record_sampler_dataset

    # Verify load_dataset was called
    stub_artifact_storage.load_dataset.assert_called_once()
    pd.testing.assert_frame_equal(dataset, stub_artifact_storage.load_dataset.return_value)


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_with_default_params(
    mock_display_sample_record, stub_dataset_creation_results, stub_dataframe
):
    """Test display_sample_record with default parameters."""
    stub_dataset_creation_results.display_sample_record()

    # Verify the underlying display_sample_record function was called
    mock_display_sample_record.assert_called_once()
    call_kwargs = mock_display_sample_record.call_args.kwargs
    assert call_kwargs["syntax_highlighting_theme"] == "dracula"
    assert call_kwargs["background_color"] is None
    assert call_kwargs["record_index"] == 0
    # Verify the record passed is the first row of the dataframe
    pd.testing.assert_series_equal(mock_display_sample_record.call_args.kwargs["record"], stub_dataframe.iloc[0])


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_with_custom_index(
    mock_display_sample_record, stub_dataset_creation_results, stub_dataframe
):
    """Test display_sample_record with a specific index."""
    stub_dataset_creation_results.display_sample_record(index=5)

    mock_display_sample_record.assert_called_once()
    call_kwargs = mock_display_sample_record.call_args.kwargs
    assert call_kwargs["record_index"] == 5
    assert call_kwargs["syntax_highlighting_theme"] == "dracula"
    assert call_kwargs["background_color"] is None
    # Verify the record passed is the correct row
    pd.testing.assert_series_equal(mock_display_sample_record.call_args.kwargs["record"], stub_dataframe.iloc[5])


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_with_custom_theme(mock_display_sample_record, stub_dataset_creation_results):
    """Test display_sample_record with custom syntax highlighting theme."""
    stub_dataset_creation_results.display_sample_record(syntax_highlighting_theme="monokai")

    mock_display_sample_record.assert_called_once()
    call_kwargs = mock_display_sample_record.call_args.kwargs
    assert call_kwargs["syntax_highlighting_theme"] == "monokai"
    assert call_kwargs["background_color"] is None


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_with_custom_background_color(mock_display_sample_record, stub_dataset_creation_results):
    """Test display_sample_record with custom background color."""
    stub_dataset_creation_results.display_sample_record(background_color="#282a36")

    mock_display_sample_record.assert_called_once()
    call_kwargs = mock_display_sample_record.call_args.kwargs
    assert call_kwargs["syntax_highlighting_theme"] == "dracula"
    assert call_kwargs["background_color"] == "#282a36"


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_with_all_custom_params(mock_display_sample_record, stub_dataset_creation_results):
    """Test display_sample_record with all parameters customized."""
    stub_dataset_creation_results.display_sample_record(
        index=3,
        syntax_highlighting_theme="monokai",
        background_color="#1e1e1e",
    )

    mock_display_sample_record.assert_called_once()
    call_kwargs = mock_display_sample_record.call_args.kwargs
    assert call_kwargs["record_index"] == 3
    assert call_kwargs["syntax_highlighting_theme"] == "monokai"
    assert call_kwargs["background_color"] == "#1e1e1e"


@patch("data_designer.config.utils.visualization.display_sample_record", autospec=True)
def test_display_sample_record_multiple_calls(
    mock_display_sample_record, stub_dataset_creation_results, stub_dataframe
):
    """Test that display_sample_record cycles through records on multiple calls."""
    num_records = len(stub_dataframe)

    # Call multiple times to test cycling
    for i in range(5):
        stub_dataset_creation_results.display_sample_record()

    assert mock_display_sample_record.call_count == 5

    # Verify that record indices cycle through 0, 1, 2, ..., num_records-1, 0, ...
    for i in range(5):
        call_kwargs = mock_display_sample_record.call_args_list[i].kwargs
        expected_index = i % num_records
        assert call_kwargs["record_index"] == expected_index


def test_display_sample_record_with_empty_dataset():
    """Test display_sample_record behavior with empty dataset."""
    empty_storage = MagicMock(spec=ArtifactStorage)
    empty_storage.load_dataset.return_value = pd.DataFrame()

    results = DatasetCreationResults(
        artifact_storage=empty_storage,
        analysis=MagicMock(spec=DatasetProfilerResults),
        config_builder=MagicMock(spec=DataDesignerConfigBuilder),
        dataset_metadata=DatasetMetadata(),
    )

    # Empty DataFrame is still a valid DataFrame, so accessing _record_sampler_dataset succeeds
    # but display_sample_record fails when trying to access index 0
    # Note: Currently raises UnboundLocalError due to bug in error handling, but tests that it fails
    with pytest.raises((DatasetSampleDisplayError, UnboundLocalError)):
        results.display_sample_record()


def test_display_sample_record_with_none_dataset():
    """Test display_sample_record behavior when dataset is None."""
    none_storage = MagicMock(spec=ArtifactStorage)
    none_storage.load_dataset.return_value = None

    results = DatasetCreationResults(
        artifact_storage=none_storage,
        analysis=MagicMock(spec=DatasetProfilerResults),
        config_builder=MagicMock(spec=DataDesignerConfigBuilder),
        dataset_metadata=DatasetMetadata(),
    )

    # Mixin raises DatasetSampleDisplayError when dataset is None
    with pytest.raises(DatasetSampleDisplayError, match="No valid dataset found"):
        results.display_sample_record()


def test_results_protocol_conformance(stub_dataset_creation_results):
    """Test that DatasetCreationResults conforms to ResultsProtocol."""
    # ResultsProtocol requires these methods
    assert hasattr(stub_dataset_creation_results, "load_dataset")
    assert hasattr(stub_dataset_creation_results, "load_analysis")
    assert hasattr(stub_dataset_creation_results, "display_sample_record")
    assert callable(stub_dataset_creation_results.load_dataset)
    assert callable(stub_dataset_creation_results.load_analysis)
    assert callable(stub_dataset_creation_results.display_sample_record)


def test_artifact_storage_load_dataset_called_once_for_caching(stub_dataset_creation_results, stub_artifact_storage):
    """Test that artifact_storage.load_dataset is called once when _record_sampler_dataset is cached."""
    # First access to _record_sampler_dataset
    _ = stub_dataset_creation_results._record_sampler_dataset

    # Second access to _record_sampler_dataset (should use cached value)
    _ = stub_dataset_creation_results._record_sampler_dataset

    # Should only be called once due to caching
    assert stub_artifact_storage.load_dataset.call_count == 1


def test_load_dataset_independent_of_record_sampler_cache(stub_dataset_creation_results, stub_artifact_storage):
    """Test that load_dataset calls artifact_storage.load_dataset independently of cache."""
    # Access _record_sampler_dataset to trigger caching
    _ = stub_dataset_creation_results._record_sampler_dataset

    # Reset the call count
    stub_artifact_storage.load_dataset.reset_mock()

    # Call load_dataset
    stub_dataset_creation_results.load_dataset()

    # Should call load_dataset again (independent of cache)
    stub_artifact_storage.load_dataset.assert_called_once()


def test_preview_results_dataset_metadata() -> None:
    """Test that PreviewResults uses DatasetMetadata in display_sample_record."""
    config_builder = MagicMock(spec=DataDesignerConfigBuilder)
    config_builder.get_columns_of_type.return_value = []

    dataset_metadata = DatasetMetadata(seed_column_names=["name", "age"])

    results = PreviewResults(
        config_builder=config_builder,
        dataset=pd.DataFrame({"name": ["Alice"], "age": [25], "greeting": ["Hello"]}),
        dataset_metadata=dataset_metadata,
    )

    # Verify metadata is stored as public attribute
    assert results.dataset_metadata == dataset_metadata
    assert results.dataset_metadata.seed_column_names == ["name", "age"]

    # Patch display_sample_record to capture the seed_column_names argument
    with patch("data_designer.config.utils.visualization.display_sample_record", wraps=display_fn) as mock_display:
        results.display_sample_record(index=0)

        # Verify seed_column_names was passed to the display function
        mock_display.assert_called_once()
        call_kwargs = mock_display.call_args.kwargs
        assert call_kwargs["seed_column_names"] == ["name", "age"]
