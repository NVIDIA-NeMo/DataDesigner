# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable
from unittest.mock import Mock

import pytest

import data_designer.lazy_heavy_imports as lazy
from data_designer.config.column_configs import ExpressionColumnConfig, ImageColumnConfig, SamplerColumnConfig
from data_designer.config.config_builder import DataDesignerConfigBuilder
from data_designer.config.processors import DropColumnsProcessorConfig, SchemaTransformProcessorConfig
from data_designer.config.record_selection import RecordSelectionConfig
from data_designer.config.sampler_params import CategorySamplerParams, SamplerType
from data_designer.engine.dataset_builders.acceptance import AcceptanceController
from data_designer.engine.dataset_builders.dataset_builder import DatasetBuilder
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner
from data_designer.engine.dataset_builders.utils.row_group_buffer import RowGroupBufferManager
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.processing.processors.schema_transform import SchemaTransformProcessor

if TYPE_CHECKING:
    import pandas as pd


def test_selection_promotes_media_before_writing_post_batch_side_artifacts(
    stub_resource_provider: Mock,
    stub_model_configs: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_builder = DataDesignerConfigBuilder(model_configs=stub_model_configs)
    config_builder.add_column(
        SamplerColumnConfig(
            name="value",
            sampler_type=SamplerType.CATEGORY,
            params=CategorySamplerParams(values=[1]),
        )
    )
    config_builder.add_column(ExpressionColumnConfig(name="keep", expr="{{ true }}", dtype="bool"))
    config_builder.add_column(ImageColumnConfig(name="image", prompt="test image", model_alias="stub-image"))
    selection_config = RecordSelectionConfig(predicate_column="keep", max_candidate_records=1)
    config_builder.with_record_selection(selection_config)
    builder = DatasetBuilder(
        data_designer_config=config_builder.build(),
        resource_provider=stub_resource_provider,
    )

    processors = [
        SchemaTransformProcessor(
            config=SchemaTransformProcessorConfig(
                name="media_schema",
                template={"copied_image": "{{ image }}"},
            ),
            resource_provider=stub_resource_provider,
        ),
        DropColumnsProcessor(
            config=DropColumnsProcessorConfig(name="drop_image", column_names=["image"]),
            resource_provider=stub_resource_provider,
        ),
    ]
    builder._processor_runner = ProcessorRunner(processors, builder.artifact_storage)
    controller = AcceptanceController(
        config=selection_config,
        target_records=1,
        buffer_size=1,
    )
    batch = controller.next_candidate_batch()
    buffer_manager = RowGroupBufferManager(builder.artifact_storage)
    buffer_manager.init_row_group(batch.row_group_id, batch.size)

    class StubScheduler:
        traces: list[Any] = []
        # The target is reached by this committed decision, so simultaneous
        # scheduler shutdown must not leave a durable terminal failure.
        early_shutdown = True
        partial_row_groups: tuple[int, ...] = ()
        first_non_retryable_error: Exception | None = None

        def __init__(
            self,
            *,
            select_dataframe: Callable[[int, int, pd.DataFrame], pd.DataFrame],
            on_finalize_row_group: Callable[[int], None],
        ) -> None:
            self._select_dataframe = select_dataframe
            self._on_finalize_row_group = on_finalize_row_group

        async def run(self) -> None:
            staged_image = (
                builder.artifact_storage.selection_media_staging_path
                / "batch_00000"
                / "images"
                / "image"
                / "accepted.png"
            )
            staged_image.parent.mkdir(parents=True, exist_ok=True)
            staged_image.write_bytes(b"accepted image")
            buffer_manager.update_cell(batch.row_group_id, 0, "keep", True)
            buffer_manager.update_cell(batch.row_group_id, 0, "image", "images/image/accepted.png")

            selected = self._select_dataframe(
                batch.row_group_id,
                batch.size,
                buffer_manager.get_dataframe(batch.row_group_id),
            )
            buffer_manager.replace_dataframe(batch.row_group_id, selected)
            processed = builder._processor_runner.run_post_batch(
                buffer_manager.get_dataframe(batch.row_group_id),
                current_batch_number=batch.row_group_id,
                strict_row_count=True,
            )
            buffer_manager.replace_dataframe(batch.row_group_id, processed)
            self._on_finalize_row_group(batch.row_group_id)

    def prepare_async_run(*_args: Any, **kwargs: Any) -> tuple[StubScheduler, RowGroupBufferManager]:
        return (
            StubScheduler(
                select_dataframe=kwargs["select_dataframe"],
                on_finalize_row_group=kwargs["on_finalize_row_group"],
            ),
            buffer_manager,
        )

    monkeypatch.setattr(builder, "_prepare_async_run", prepare_async_run)
    stub_resource_provider.model_registry.get_model_usage_snapshot.return_value = {}
    stub_resource_provider.model_registry.get_usage_deltas.return_value = {}

    builder._run_candidate_batch(
        [],
        controller=controller,
        batch=batch,
        on_batch_complete=None,
    )

    committed_image = "images/selection_batch_00000/image/accepted.png"
    assert (builder.artifact_storage.base_dataset_path / committed_image).read_bytes() == b"accepted image"
    transformed = lazy.pd.read_parquet(
        builder.artifact_storage.processors_outputs_path / "media_schema" / "batch_00000.parquet"
    )
    assert transformed.loc[0, "copied_image"] == committed_image
    dropped = lazy.pd.read_parquet(builder.artifact_storage.dropped_columns_dataset_path / "batch_00000.parquet")
    assert dropped.loc[0, "image"] == committed_image
    accepted = lazy.pd.read_parquet(builder.artifact_storage.selection_partition_path(0))
    assert accepted.columns.tolist() == ["keep"]
    assert controller.has_reached_target
    assert controller.first_non_retryable_error is None
    marker = builder.artifact_storage.read_selection_checkpoints()[0]
    assert marker["schema_materialized"] is True
    assert marker["non_retryable_error"] is None
    assert marker["stopped_early"] is False
