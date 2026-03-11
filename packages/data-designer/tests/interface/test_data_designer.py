# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

import data_designer.interface.data_designer as dd_mod
import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.errors import InvalidConfigError
from data_designer.config.models import ModelProvider
from data_designer.config.processors import DropColumnsProcessorConfig
from data_designer.config.run_config import RunConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.config.seed_source import DirectoryListingTransform, DirectorySeedSource, HuggingFaceSeedSource
from data_designer.engine.secret_resolver import CompositeResolver, EnvironmentResolver, PlaintextResolver
from data_designer.engine.testing.stubs import StubHuggingFaceSeedReader
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.errors import DataDesignerGenerationError, DataDesignerProfilingError


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


@pytest.fixture
def stub_seed_reader():
    return StubHuggingFaceSeedReader()


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


def test_run_config_setting_persists(stub_artifact_path, stub_model_providers):
    """Test that run config setting persists across multiple calls."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)

    # Test default values
    assert data_designer._run_config.disable_early_shutdown is False
    assert data_designer._run_config.shutdown_error_rate == 0.5
    assert data_designer._run_config.shutdown_error_window == 10
    assert data_designer._run_config.buffer_size == 1000
    assert data_designer._run_config.max_conversation_restarts == 5
    assert data_designer._run_config.max_conversation_correction_steps == 0

    # Test setting custom values
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=True,
            shutdown_error_rate=0.8,
            shutdown_error_window=25,
            buffer_size=500,
            max_conversation_restarts=7,
            max_conversation_correction_steps=2,
        )
    )
    assert data_designer._run_config.disable_early_shutdown is True
    assert data_designer._run_config.shutdown_error_rate == 1.0  # normalized when disabled
    assert data_designer._run_config.shutdown_error_window == 25
    assert data_designer._run_config.buffer_size == 500
    assert data_designer._run_config.max_conversation_restarts == 7
    assert data_designer._run_config.max_conversation_correction_steps == 2

    # Test updating values
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=False,
            shutdown_error_rate=0.3,
            shutdown_error_window=5,
            buffer_size=750,
            max_conversation_restarts=9,
            max_conversation_correction_steps=1,
        )
    )
    assert data_designer._run_config.disable_early_shutdown is False
    assert data_designer._run_config.shutdown_error_rate == 0.3
    assert data_designer._run_config.shutdown_error_window == 5
    assert data_designer._run_config.buffer_size == 750
    assert data_designer._run_config.max_conversation_restarts == 9
    assert data_designer._run_config.max_conversation_correction_steps == 1


def test_run_config_normalizes_error_rate_when_disabled(stub_artifact_path, stub_model_providers):
    """Test that shutdown_error_rate is normalized to 1.0 when disabled."""
    data_designer = DataDesigner(artifact_path=stub_artifact_path, model_providers=stub_model_providers)

    # When enabled (default), shutdown_error_rate should use the configured value
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=False,
            shutdown_error_rate=0.7,
        )
    )
    assert data_designer._run_config.shutdown_error_rate == 0.7

    # When disabled, shutdown_error_rate should be normalized to 1.0
    data_designer.set_run_config(
        RunConfig(
            disable_early_shutdown=True,
            shutdown_error_rate=0.7,
        )
    )
    assert data_designer._run_config.shutdown_error_rate == 1.0


def test_run_config_rejects_invalid_buffer_size() -> None:
    with pytest.raises(ValidationError, match="buffer_size"):
        RunConfig(buffer_size=0)


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
        mock_builder.artifact_storage.load_dataset_with_dropped_columns.return_value = lazy.pd.DataFrame(
            {"col": [1, 2, 3]}
        )
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
        mock_builder.build_preview.return_value = lazy.pd.DataFrame({"col": [1, 2, 3]})
        mock_builder.process_preview.return_value = lazy.pd.DataFrame({"col": [1, 2, 3]})
        mock_builder_method.return_value = mock_builder

        # Mock profiler to fail
        mock_profiler = MagicMock()
        mock_profiler.profile_dataset.side_effect = ValueError("Profiler failed in preview")
        mock_profiler_method.return_value = mock_profiler

        with pytest.raises(
            DataDesignerProfilingError, match="🛑 Error profiling preview dataset: Profiler failed in preview"
        ):
            data_designer.preview(stub_sampler_only_config_builder, num_records=3)


def test_create_raises_generation_error_when_dataset_is_empty(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """When all records are dropped during generation, create should raise
    DataDesignerGenerationError with a clear message instead of a misleading profiler error.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.storage.artifact_storage.ArtifactStorage.load_dataset_with_dropped_columns",
        return_value=lazy.pd.DataFrame(),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Dataset is empty"):
            data_designer.create(stub_sampler_only_config_builder, num_records=1)


def test_create_raises_generation_error_when_load_dataset_fails(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_sampler_only_config_builder: DataDesignerConfigBuilder,
    stub_managed_assets_path: Path,
) -> None:
    """When no parquet was written (e.g. all records dropped), load_dataset_with_dropped_columns
    raises an exception. create() should surface this as DataDesignerGenerationError, not
    DataDesignerProfilingError.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.storage.artifact_storage.ArtifactStorage.load_dataset_with_dropped_columns",
        side_effect=FileNotFoundError("No parquet files found"),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Failed to load generated dataset"):
            data_designer.create(stub_sampler_only_config_builder, num_records=1)


def test_preview_raises_generation_error_when_dataset_is_empty(
    stub_artifact_path, stub_model_providers, stub_sampler_only_config_builder, stub_managed_assets_path
):
    """When all records are dropped during generation, preview should raise
    DataDesignerGenerationError with a clear message instead of a misleading profiler error.
    """
    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    with patch(
        "data_designer.engine.dataset_builders.column_wise_builder.ColumnWiseDatasetBuilder.process_preview",
        return_value=lazy.pd.DataFrame(),
    ):
        with pytest.raises(DataDesignerGenerationError, match="Dataset is empty"):
            data_designer.preview(stub_sampler_only_config_builder, num_records=1)


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

    config_builder.add_processor(DropColumnsProcessorConfig(name="drop_columns_processor", column_names=["category"]))

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


def test_validate_raises_error_when_seed_collides(
    stub_artifact_path,
    stub_model_providers,
    stub_model_configs,
    stub_managed_assets_path,
    stub_seed_reader,
):
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.with_seed_dataset(HuggingFaceSeedSource(path="hf://datasets/test/data.csv"))
    config_builder.add_column(
        SamplerColumnConfig(
            name="city",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=["new york", "los angeles"]),
        )
    )

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
        seed_readers=[stub_seed_reader],
    )

    with pytest.raises(InvalidConfigError):
        data_designer.validate(config_builder)


def test_initialize_interface_runtime_runs_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """_initialize_interface_runtime only runs initialization once."""
    monkeypatch.setattr(dd_mod, "_interface_runtime_initialized", False)

    with (
        patch("data_designer.interface.data_designer.configure_logging") as mock_logging,
        patch("data_designer.interface.data_designer.resolve_seed_default_model_settings") as mock_resolve,
    ):
        dd_mod._initialize_interface_runtime()
        dd_mod._initialize_interface_runtime()
        mock_logging.assert_called_once()
        mock_resolve.assert_called_once()


def test_create_dataset_e2e_with_directory_seed_source_without_transform(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-seed"
    (seed_dir / "subdir").mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "subdir" / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(DirectorySeedSource(path=str(seed_dir), file_pattern="*.txt"))
    builder.add_column(ExpressionColumnConfig(name="path_label", expr="{{ source_kind }}::{{ relative_path }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=2, dataset_name="directory-seed-test")
    df = results.load_dataset().sort_values("relative_path").reset_index(drop=True)

    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["relative_path"]) == ["alpha.txt", "subdir/beta.txt"]
    assert list(df["file_name"]) == ["alpha.txt", "beta.txt"]
    assert list(df["path_label"]) == [
        "directory_file::alpha.txt",
        "directory_file::subdir/beta.txt",
    ]


def test_create_dataset_e2e_with_directory_seed_source_transform(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-seed-transform"
    (seed_dir / "subdir").mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "subdir" / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        DirectorySeedSource(
            path=str(seed_dir),
            file_pattern="*.txt",
            transform=DirectoryListingTransform(),
        )
    )
    builder.add_column(ExpressionColumnConfig(name="path_label", expr="{{ source_kind }}::{{ relative_path }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=2, dataset_name="directory-seed-transform-test")
    df = results.load_dataset().sort_values("relative_path").reset_index(drop=True)

    assert list(df["source_kind"]) == ["directory_file", "directory_file"]
    assert list(df["relative_path"]) == ["alpha.txt", "subdir/beta.txt"]
    assert list(df["path_label"]) == [
        "directory_file::alpha.txt",
        "directory_file::subdir/beta.txt",
    ]


def test_create_dataset_e2e_with_directory_seed_source_non_recursive(
    stub_artifact_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_managed_assets_path: Path,
    tmp_path: Path,
) -> None:
    seed_dir = tmp_path / "directory-seed-non-recursive"
    (seed_dir / "subdir").mkdir(parents=True)
    (seed_dir / "alpha.txt").write_text("alpha", encoding="utf-8")
    (seed_dir / "subdir" / "beta.txt").write_text("beta", encoding="utf-8")

    builder = DataDesignerConfigBuilder()
    builder.with_seed_dataset(
        DirectorySeedSource(
            path=str(seed_dir),
            file_pattern="*.txt",
            recursive=False,
        )
    )
    builder.add_column(ExpressionColumnConfig(name="path_label", expr="{{ source_kind }}::{{ relative_path }}"))

    data_designer = DataDesigner(
        artifact_path=stub_artifact_path,
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
        managed_assets_path=stub_managed_assets_path,
    )

    results = data_designer.create(builder, num_records=1, dataset_name="directory-seed-non-recursive-test")
    df = results.load_dataset().sort_values("relative_path").reset_index(drop=True)

    assert list(df["source_kind"]) == ["directory_file"]
    assert list(df["relative_path"]) == ["alpha.txt"]
    assert list(df["path_label"]) == ["directory_file::alpha.txt"]
