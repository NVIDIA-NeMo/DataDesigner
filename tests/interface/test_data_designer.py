# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from data_designer.config.column_configs import SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.errors import InvalidFileFormatError
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.seed import LocalSeedDatasetReference
from data_designer.engine.model_provider import ModelProvider
from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import (
    DataDesignerGenerationError,
    DataDesignerProfilingError,
    InvalidBufferValueError,
)


@pytest.fixture
def stub_artifact_path(tmp_path):
    """Temporary directory for artifacts."""
    return tmp_path / "artifacts"


@pytest.fixture
def stub_managed_assets_path(tmp_path):
    """Temporary directory for managed assets."""
    managed_path = tmp_path / "managed-assets"
    managed_path.mkdir(parents=True, exist_ok=True)
    return managed_path


@pytest.fixture
def stub_model_providers():
    return [
        ModelProvider(
            name="stub-model-provider",
            endpoint="https://api.stub-model-provider.com/v1",
            api_key="stub-model-provider-api-key",
        )
    ]


def test_init_with_custom_secret_resolver(stub_artifact_path, stub_model_providers):
    """Test DataDesigner initialization with custom secret resolver."""
    designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
    )
    assert designer is not None


def test_init_with_default_composite_secret_resolver(stub_artifact_path, stub_model_providers):
    """Test DataDesigner initialization with default composite secret resolver."""
    designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    assert designer is not None
    assert isinstance(designer.secret_resolver, CompositeResolver)
    # Verify the composite resolver is properly configured with the expected resolvers
    resolvers = designer.secret_resolver.resolvers
    assert len(resolvers) == 2
    assert isinstance(resolvers[0], EnvironmentResolver)
    assert isinstance(resolvers[1], PlaintextResolver)


def test_init_with_string_path(stub_artifact_path, stub_model_providers):
    """Test DataDesigner accepts string paths."""
    designer = DataDesigner(artifact_path=str(stub_artifact_path), model_providers=stub_model_providers)
    assert designer is not None
    assert isinstance(designer._artifact_path, Path)


def test_init_with_path_object(stub_artifact_path, stub_model_providers):
    """Test DataDesigner accepts Path objects."""
    designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    assert designer is not None


def test_make_seed_reference_from_dataframe(stub_dataframe):
    """Test creating seed reference from DataFrame."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.parquet"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=file_path)

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)
        assert file_path.exists()

        # Verify the file contains the correct data
        loaded_df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(loaded_df, stub_dataframe)


def test_make_seed_reference_from_dataframe_writes_parquet_format(stub_dataframe):
    """Test that seed reference writes DataFrame as parquet."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.parquet"
        DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=file_path)

        # Verify we can read it back as parquet
        loaded_df = pd.read_parquet(file_path)
        assert len(loaded_df) == len(stub_dataframe)
        assert list(loaded_df.columns) == list(stub_dataframe.columns)


def test_make_seed_reference_from_dataframe_writes_csv_format(stub_dataframe):
    """Test that seed reference writes DataFrame as CSV."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.csv"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=file_path)

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)
        assert file_path.exists()

        # Verify we can read it back as CSV
        loaded_df = pd.read_csv(file_path)
        assert len(loaded_df) == len(stub_dataframe)
        assert list(loaded_df.columns) == list(stub_dataframe.columns)


def test_make_seed_reference_from_dataframe_writes_json_format(stub_dataframe):
    """Test that seed reference writes DataFrame as JSON."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.json"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=file_path)

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)
        assert file_path.exists()

        # Verify we can read it back as JSON
        loaded_df = pd.read_json(file_path, orient="records", lines=True)
        assert len(loaded_df) == len(stub_dataframe)
        assert list(loaded_df.columns) == list(stub_dataframe.columns)


def test_make_seed_reference_from_dataframe_writes_jsonl_format(stub_dataframe):
    """Test that seed reference writes DataFrame as JSONL."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.jsonl"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=file_path)

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)
        assert file_path.exists()

        # Verify we can read it back as JSONL
        loaded_df = pd.read_json(file_path, orient="records", lines=True)
        assert len(loaded_df) == len(stub_dataframe)
        assert list(loaded_df.columns) == list(stub_dataframe.columns)


def test_make_seed_reference_from_dataframe_raises_error_for_invalid_extension(stub_dataframe):
    """Test that make_seed_reference_from_dataframe raises error for invalid file extensions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test with .txt extension
        txt_path = Path(temp_dir) / "seed.txt"
        with pytest.raises(InvalidFileFormatError):
            DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=txt_path)

        # Test with no extension
        no_ext_path = Path(temp_dir) / "seed"
        with pytest.raises(InvalidFileFormatError):
            DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=no_ext_path)


def test_make_seed_reference_from_dataframe_accepts_uppercase_extensions(stub_dataframe):
    """Test that make_seed_reference_from_dataframe accepts uppercase file extensions (case insensitive)."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Test .PARQUET
        parquet_path = Path(temp_dir) / "seed.PARQUET"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=parquet_path)
        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(parquet_path)
        assert parquet_path.exists()

        # Test .CSV
        csv_path = Path(temp_dir) / "seed.CSV"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=csv_path)
        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(csv_path)
        assert csv_path.exists()

        # Test .JSON
        json_path = Path(temp_dir) / "seed.JSON"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=json_path)
        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(json_path)
        assert json_path.exists()

        # Test .JSONL
        jsonl_path = Path(temp_dir) / "seed.JSONL"
        ref = DataDesigner.make_seed_reference_from_dataframe(stub_dataframe, file_path=jsonl_path)
        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(jsonl_path)
        assert jsonl_path.exists()


def test_make_seed_reference_from_dataframe_overwrites_existing_file(stub_dataframe):
    """Test that make_seed_reference_from_dataframe overwrites existing file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "seed.parquet"

        # Create initial file with different data
        initial_df = pd.DataFrame({"col": [999]})
        initial_df.to_parquet(file_path)

        # Overwrite with new data
        new_df = pd.DataFrame({"col": [1, 2, 3]})
        _ = DataDesigner.make_seed_reference_from_dataframe(new_df, file_path=file_path)

        # Verify the file was overwritten
        loaded_df = pd.read_parquet(file_path)
        pd.testing.assert_frame_equal(loaded_df, new_df)
        assert len(loaded_df) == 3  # Should have 3 rows, not 1


def test_make_seed_reference_from_file_with_string_path():
    """Test creating seed reference from file path string."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "dataset.parquet"
        df = pd.DataFrame({"col": [1, 2, 3]})
        df.to_parquet(file_path)

        ref = DataDesigner.make_seed_reference_from_file(str(file_path))

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)


def test_make_seed_reference_from_file_with_path_object(stub_dataframe):
    """Test creating seed reference from Path object."""
    with tempfile.TemporaryDirectory() as temp_dir:
        file_path = Path(temp_dir) / "dataset.parquet"
        stub_dataframe.to_parquet(file_path)

        ref = DataDesigner.make_seed_reference_from_file(file_path)

        assert isinstance(ref, LocalSeedDatasetReference)
        assert ref.dataset == str(file_path)


def test_buffer_size_setting_persists(stub_artifact_path, stub_model_providers):
    """Test that buffer size setting persists across multiple calls."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    custom_buffer_size = 500
    data_designer.set_buffer_size(custom_buffer_size)
    assert data_designer._buffer_size == custom_buffer_size

    another_buffer_size = 750
    data_designer.set_buffer_size(another_buffer_size)
    assert data_designer._buffer_size == another_buffer_size


def test_set_buffer_size_raises_error_for_invalid_buffer_size(stub_artifact_path, stub_model_providers):
    """Test that set_buffer_size raises error for invalid buffer size."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)
    with pytest.raises(InvalidBufferValueError, match="Buffer size must be greater than 0."):
        data_designer.set_buffer_size(0)


def test_multiple_seed_references_can_be_created():
    """Test that multiple seed references can be created from different sources."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create seed reference from DataFrame
        df1 = pd.DataFrame({"col": [1, 2, 3]})
        file_path_1 = Path(temp_dir) / "seed1.parquet"
        ref1 = DataDesigner.make_seed_reference_from_dataframe(df1, file_path=file_path_1)

        # Create seed reference from another DataFrame
        df2 = pd.DataFrame({"col": [4, 5, 6]})
        file_path_2 = Path(temp_dir) / "seed2.parquet"
        ref2 = DataDesigner.make_seed_reference_from_dataframe(df2, file_path=file_path_2)

        # Create seed reference from existing file
        ref3 = DataDesigner.make_seed_reference_from_file(file_path_1)

        # Verify all references are unique and valid
        assert ref1.dataset != ref2.dataset
        assert ref1.dataset == ref3.dataset
        assert all(isinstance(ref, LocalSeedDatasetReference) for ref in [ref1, ref2, ref3])


def test_create_dataset_e2e_using_only_sampler_columns(
    stub_sampler_only_config_builder, stub_artifact_path, stub_model_providers, stub_managed_assets_path
):
    column_names = [config.name for config in stub_sampler_only_config_builder.get_column_configs()]

    num_records = 3

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(stub_sampler_only_config_builder, num_records=num_records)

    df = results.load_dataset()
    assert len(df) == num_records
    assert set(df.columns) == set(column_names)

    # cycle through with no errors
    for _ in range(num_records + 2):
        results.display_sample_record()

    analysis = results.load_analysis()
    assert analysis.target_num_records == num_records

    # display report with no errors
    analysis.to_report()


def test_create_raises_error_when_builder_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that create method raises DataDesignerCreateError when builder.build fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch.object(data_designer, "_create_dataset_builder") as mock_builder_method:
        mock_builder = MagicMock()
        mock_builder.build.side_effect = RuntimeError("Builder failed")
        mock_builder_method.return_value = mock_builder

        with pytest.raises(DataDesignerGenerationError, match="🛑 Error generating dataset: Builder failed"):
            data_designer.create(stub_sampler_only_config_builder, num_records=3)


def test_create_raises_error_when_profiler_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that create method raises DataDesignerCreateError when profiler.profile_dataset fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with (
        patch.object(data_designer, "_create_dataset_builder") as mock_builder_method,
        patch.object(data_designer, "_create_dataset_profiler") as mock_profiler_method,
    ):
        # Mock builder to succeed
        mock_builder = MagicMock()
        mock_builder.build.return_value = None
        mock_builder.artifact_storage.load_dataset_with_dropped_columns.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_builder_method.return_value = mock_builder

        # Mock profiler to fail
        mock_profiler = MagicMock()
        mock_profiler.profile_dataset.side_effect = ValueError("Profiler failed")
        mock_profiler_method.return_value = mock_profiler

        with pytest.raises(DataDesignerProfilingError, match="🛑 Error profiling dataset: Profiler failed"):
            data_designer.create(stub_sampler_only_config_builder, num_records=3)


def test_preview_raises_error_when_builder_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that preview method raises DataDesignerPreviewError when builder.build_preview fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch.object(data_designer, "_create_dataset_builder") as mock_builder_method:
        mock_builder = MagicMock()
        mock_builder.build_preview.side_effect = RuntimeError("Builder preview failed")
        mock_builder_method.return_value = mock_builder

        with pytest.raises(
            DataDesignerGenerationError, match="🛑 Error generating preview dataset: Builder preview failed"
        ):
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)


def test_preview_raises_error_when_profiler_fails(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """Test that preview method raises DataDesignerPreviewError when profiler.profile_dataset fails."""
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with (
        patch.object(data_designer, "_create_dataset_builder") as mock_builder_method,
        patch.object(data_designer, "_create_dataset_profiler") as mock_profiler_method,
    ):
        # Mock builder to succeed
        mock_builder = MagicMock()
        mock_builder.build_preview.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_builder.process_preview.return_value = pd.DataFrame({"col": [1, 2, 3]})
        mock_builder_method.return_value = mock_builder

        # Mock profiler to fail
        mock_profiler = MagicMock()
        mock_profiler.profile_dataset.side_effect = ValueError("Profiler failed in preview")
        mock_profiler_method.return_value = mock_profiler

        with pytest.raises(
            DataDesignerProfilingError, match="🛑 Error profiling preview dataset: Profiler failed in preview"
        ):
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)


def test_preview_with_dropped_columns(
    stub_artifact_path, stub_model_providers, stub_model_configs, stub_managed_assets_path
):
    """Test that preview correctly handles dropped columns and maintains consistency."""
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(
            name="uuid", sampler_type="uuid", params={"prefix": "id_", "short_form": True, "uppercase": False}
        )
    )
    config_builder.add_column(
        SamplerColumnConfig(name="category", sampler_type="category", params={"values": ["a", "b", "c"]})
    )
    config_builder.add_column(
        SamplerColumnConfig(name="uniform", sampler_type="uniform", params={"low": 1, "high": 100})
    )

    config_builder.add_processor(
        DropColumnsProcessorConfig(
            name="drop_columns_processor", build_stage=BuildStage.POST_BATCH, column_names=["category"]
        )
    )

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    num_records = 5
    preview_results = data_designer.preview(config_builder, num_records=num_records)

    preview_dataset = preview_results.dataset

    assert "category" not in preview_dataset.columns, "Dropped column 'category' should not be in preview dataset"

    assert "uuid" in preview_dataset.columns, "Column 'uuid' should be in preview dataset"
    assert "uniform" in preview_dataset.columns, "Column 'uniform' should be in preview dataset"

    assert len(preview_dataset) == num_records, f"Preview dataset should have {num_records} records"

    analysis = preview_results.analysis
    assert analysis is not None, "Analysis should be generated"

    column_names_in_analysis = [stat.column_name for stat in analysis.column_statistics]
    assert "uuid" in column_names_in_analysis, "Column 'uuid' should be in analysis"
    assert "uniform" in column_names_in_analysis, "Column 'uniform' should be in analysis"
    assert "category" not in column_names_in_analysis, "Dropped column 'category' should not be in analysis statistics"

    assert analysis.side_effect_column_names is not None, "Side effect column names should be tracked"
    assert "category" in analysis.side_effect_column_names, (
        "Dropped column 'category' should be tracked in side_effect_column_names"
    )
