# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pyarrow as pa
from pytest import fixture

from data_designer.config.analysis.column_statistics import (
    CategoricalHistogramData,
    ColumnDistributionType,
    NumericalDistribution,
)
from data_designer.config.column_configs import LLMJudgeColumnConfig, Score
from data_designer.config.column_types import ColumnConfigT
from data_designer.config.models import ModelConfig
from data_designer.engine.analysis.dataset_profiler import (
    DataDesignerDatasetProfiler,
    DatasetProfilerConfig,
)
from data_designer.engine.analysis.utils.judge_score_processing import JudgeScoreDistributions
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage
from data_designer.engine.models.registry import ModelRegistry
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider


@fixture
def test_data_path() -> Path:
    return Path(__file__).parent / "test_data"


@fixture
def stub_artifact_path(test_data_path: Path) -> Path:
    return test_data_path / "artifacts"


@fixture
def stub_dataset_path(stub_artifact_path: Path) -> Path:
    return stub_artifact_path / "dataset"


@fixture
def stub_df(stub_dataset_path: Path) -> pd.DataFrame:
    return pd.read_json(
        stub_dataset_path / "dataset.json",
        orient="records",
        dtype_backend="pyarrow",
    )


@fixture
def stub_dataset_metadata_path(stub_dataset_path: Path) -> Path:
    return stub_dataset_path / "metadata.json"


@fixture
def column_configs(dataset_profiler: DataDesignerDatasetProfiler) -> list[ColumnConfigT]:
    return dataset_profiler.config.column_configs


@fixture
def dataset_profiler(
    stub_dataset_path: Path,
    artifact_storage: ArtifactStorage,
) -> DataDesignerDatasetProfiler:
    # Ensure the final dataset path exists
    with open(stub_dataset_path / "column_configs.json", "r") as f:
        column_configs = json.load(f)

    model_config = Mock(spec=ModelConfig)
    model_config.alias = "nano"

    model_registry = Mock(spec=ModelRegistry)
    model_registry.model_configs = {"nano": model_config}

    profiler = DataDesignerDatasetProfiler(
        config=DatasetProfilerConfig(column_configs=column_configs),
        resource_provider=ResourceProvider(artifact_storage=artifact_storage, model_registry=model_registry),
    )

    return profiler


@fixture
def stub_df_with_mixed_column_types():
    data = {
        "int_column": [1, 2, 3, 4, 5],
        "float_column": [1.1, 2.2, 3.3, 4.4, 5.5],
        "string_column": ["a", "b", "c", "d", "e"],
        "int_with_nulls_column": [1, 2, None, 4, None],
    }
    return pa.Table.from_pydict(data).to_pandas(types_mapper=pd.ArrowDtype)


@fixture
def mock_prompt_renderer_render():
    with patch(
        "data_designer.engine.analysis.utils.column_statistics_calculations.RecordBasedPromptRenderer.render"
    ) as mock:
        yield mock


@fixture
def data_designer_registry() -> DataDesignerRegistry:
    return DataDesignerRegistry()


@fixture
def stub_score():
    """Create a sample rubric for testing."""
    return Score(
        name="Quality",
        description="Quality assessment score",
        options={
            4: "Excellent quality",
            3: "Good quality",
            2: "Fair quality",
            1: "Poor quality",
            0: "Very poor quality",
        },
    )


@fixture
def stub_judge_column_config(stub_score):
    """Create a sample LLMJudgeColumnConfig for testing."""
    return LLMJudgeColumnConfig(
        name="judge_scores",
        prompt="Evaluate the quality",
        model_alias="test_model",
        scores=[stub_score],
    )


@fixture
def stub_judge_distributions():
    return JudgeScoreDistributions(
        scores={"Quality": [4, 3, 2, 1, 0]},
        reasoning={"Quality": ["Excellent", "Good", "Fair", "Poor", "Very Poor"]},
        distribution_types={"Quality": ColumnDistributionType.NUMERICAL},
        distributions={"Quality": NumericalDistribution(min=0, max=4, mean=2.0, stddev=1.4, median=2.0)},
        histograms={"Quality": CategoricalHistogramData(categories=[4, 3, 2, 1, 0], counts=[1, 1, 1, 1, 1])},
    )


@fixture
def stub_resource_provider_no_model_registry(tmp_path):
    """Create a mock ResourceProvider for testing."""
    return ResourceProvider(artifact_storage=ArtifactStorage(artifact_path=tmp_path))
