# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import threading
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


def _creation_result(
    config_builder: DataDesignerConfigBuilder,
    stub_dataset_profiler_results,
) -> DatasetCreationResults:
    return DatasetCreationResults(
        artifact_storage=MagicMock(),
        analysis=stub_dataset_profiler_results,
        config_builder=config_builder,
        dataset_metadata=DatasetMetadata(),
    )


@pytest.mark.asyncio
async def test_acreate_delegates_to_create(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", model_providers=stub_model_providers)
    artifact_storage = MagicMock()
    expected = _creation_result(_seeded_builder(stub_model_configs, ["Ada"]), stub_dataset_profiler_results)
    expected.artifact_storage = artifact_storage
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
async def test_acreate_does_not_serialize_create_calls(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
    stub_dataset_profiler_results,
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", model_providers=stub_model_providers)
    started_count = 0
    started_lock = threading.Lock()
    both_started = threading.Event()
    release = threading.Event()

    def fake_create(
        config_builder: DataDesignerConfigBuilder,
        *,
        num_records: int,
        dataset_name: str,
        resume: ResumeMode = ResumeMode.NEVER,
    ) -> DatasetCreationResults:
        nonlocal started_count
        del num_records, dataset_name, resume
        with started_lock:
            started_count += 1
            if started_count == 2:
                both_started.set()
        assert both_started.wait(5)
        assert release.wait(5)
        return _creation_result(config_builder, stub_dataset_profiler_results)

    data_designer.create = MagicMock(side_effect=fake_create)
    left = _seeded_builder(stub_model_configs, ["Ada"])
    right = _seeded_builder(stub_model_configs, ["Grace"])

    left_task = asyncio.create_task(data_designer.acreate(left, num_records=1, dataset_name="left"))
    right_task = asyncio.create_task(data_designer.acreate(right, num_records=1, dataset_name="right"))
    try:
        assert await asyncio.to_thread(both_started.wait, 5)
    finally:
        release.set()

    left_result, right_result = await asyncio.gather(left_task, right_task)

    assert isinstance(left_result, DatasetCreationResults)
    assert isinstance(right_result, DatasetCreationResults)
    assert data_designer.create.call_count == 2


def test_data_designer_reuses_request_admission_across_create_calls(
    tmp_path: Path,
    stub_model_providers: list[ModelProvider],
    stub_model_configs: list[ModelConfig],
) -> None:
    data_designer = DataDesigner(artifact_path=tmp_path / "artifacts", model_providers=stub_model_providers)
    left = _seeded_builder(stub_model_configs, ["Ada"])
    right = _seeded_builder(stub_model_configs, ["Grace"])

    left_provider = data_designer._create_resource_provider("left", left)
    right_provider = data_designer._create_resource_provider("right", right)

    assert left_provider.model_registry is not None
    assert right_provider.model_registry is not None
    assert left_provider.model_registry.request_admission is right_provider.model_registry.request_admission
    assert left_provider.model_registry.request_admission is data_designer._request_admission


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
