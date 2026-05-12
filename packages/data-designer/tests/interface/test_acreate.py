# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.dataset_metadata import DatasetMetadata
from data_designer.config.models import ModelConfig, ModelProvider
from data_designer.config.seed_source_dataframe import DataFrameSeedSource
from data_designer.engine.secret_resolver import PlaintextResolver
from data_designer.engine.storage.artifact_storage import ResumeMode
from data_designer.interface.data_designer import DataDesigner
from data_designer.interface.results import DatasetCreationResults


def _seeded_builder(model_configs: list[ModelConfig], names: list[str]) -> DataDesignerConfigBuilder:
    builder = DataDesignerConfigBuilder(model_configs=model_configs)
    builder.with_seed_dataset(DataFrameSeedSource(df=lazy.pd.DataFrame({"name": names})))
    builder.add_column(ExpressionColumnConfig(name="name_copy", expr="{{ name }}"))
    return builder


@pytest.mark.asyncio
async def test_acreate_delegates_to_create(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", model_providers=stub_model_providers)
    artifact_storage = MagicMock()
    expected = DatasetCreationResults(
        artifact_storage=artifact_storage,
        analysis=stub_dataset_profiler_results,
        config_builder=_seeded_builder(stub_model_configs, ["Ada"]),
        dataset_metadata=DatasetMetadata(),
    )
    data_designer.create = MagicMock(return_value=expected)
    builder = _seeded_builder(stub_model_configs, ["Ada"])

    result = await data_designer.acreate(
        builder,
        num_records=1,
        dataset_name="async-dataset",
        resume=ResumeMode.IF_POSSIBLE,
    )

    assert result is expected
    data_designer.create.assert_called_once_with(
        builder,
        num_records=1,
        dataset_name="async-dataset",
        resume=ResumeMode.IF_POSSIBLE,
    )


@pytest.mark.asyncio
async def test_acreate_supports_gathered_real_async_workflows(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    data_designer = DataDesigner(
        artifact_path=tmp_path / "artifacts",
        model_providers=stub_model_providers,
        secret_resolver=PlaintextResolver(),
    )
    left = _seeded_builder(stub_model_configs, ["Ada", "Linus"])
    right = _seeded_builder(stub_model_configs, ["Grace"])

    left_result, right_result = await asyncio.gather(
        data_designer.acreate(left, num_records=2, dataset_name="left"),
        data_designer.acreate(right, num_records=1, dataset_name="right"),
    )

    assert left_result.load_dataset().sort_values("name")["name_copy"].tolist() == ["Ada", "Linus"]
    assert right_result.load_dataset()["name_copy"].tolist() == ["Grace"]
