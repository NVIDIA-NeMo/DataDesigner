# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import logging
import os
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from data_designer.config.column_configs import CustomColumnConfig
from data_designer.config.column_types import ColumnConfigT, DataDesignerColumnType
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
)
from data_designer.config.version import get_library_version
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorWithModel,
    GenerationStrategy,
)
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
from data_designer.engine.dataset_builders.utils.processor_runner import ProcessorRunner
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum, TelemetryHandler
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.engine.storage.artifact_storage import SDG_CONFIG_FILENAME, ArtifactStorage
from data_designer.engine.storage.media_storage import StorageMode
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
    from data_designer.engine.models.usage import ModelUsageStats

logger = logging.getLogger(__name__)

DATA_DESIGNER_ASYNC_ENGINE = os.environ.get("DATA_DESIGNER_ASYNC_ENGINE", "0") == "1"

if DATA_DESIGNER_ASYNC_ENGINE:
    import sys

    if sys.version_info < (3, 11):
        raise RuntimeError(
            "DATA_DESIGNER_ASYNC_ENGINE requires Python 3.11+ (asyncio.TaskGroup). "
            f"Current version: {sys.version_info.major}.{sys.version_info.minor}"
        )
    from data_designer.engine.dataset_builders.utils.async_concurrency import AsyncConcurrentExecutor

    logger.info("⚡ DATA_DESIGNER_ASYNC_ENGINE is enabled — using async concurrency")

_CLIENT_VERSION: str = get_library_version()


class ColumnWiseDatasetBuilder:
    def __init__(
        self,
        data_designer_config: DataDesignerConfig,
        resource_provider: ResourceProvider,
        registry: DataDesignerRegistry | None = None,
    ):
        self.batch_manager = DatasetBatchManager(resource_provider.artifact_storage)
        self._resource_provider = resource_provider
        self._records_to_drop: set[int] = set()
        self._registry = registry or DataDesignerRegistry()

        self._data_designer_config = compile_data_designer_config(data_designer_config, resource_provider)
        self._column_configs = compile_dataset_builder_column_configs(self._data_designer_config)
        processors = self._initialize_processors(self._data_designer_config.processors or [])
        self._processor_runner = ProcessorRunner(
            processors=processors,
            artifact_storage=resource_provider.artifact_storage,
        )
        self._validate_column_configs()

    @property
    def artifact_storage(self) -> ArtifactStorage:
        return self._resource_provider.artifact_storage

    @property
    def processors(self) -> tuple[Processor, ...]:
        return self._processor_runner.processors

    def set_processor_runner(self, processors: list[Processor]) -> None:
        """Replace the processor runner with a new one using the given processors."""
        self._processor_runner = ProcessorRunner(
            processors=processors,
            artifact_storage=self.artifact_storage,
        )

    @functools.cached_property
    def single_column_configs(self) -> list[ColumnConfigT]:
        configs = []
        for config in self._column_configs:
            if isinstance(config, MultiColumnConfig):
                configs.extend(config.columns)
            else:
                configs.append(config)
        return configs

    @functools.cached_property
    def llm_generated_column_configs(self) -> list[ColumnConfigT]:
        return [config for config in self.single_column_configs if column_type_is_model_generated(config.column_type)]

    def build(
        self,
        *,
        num_records: int,
        on_batch_complete: Callable[[Path], None] | None = None,
        save_multimedia_to_disk: bool = True,
    ) -> Path:
        """Build the dataset.

        Args:
            num_records: Number of records to generate.
            on_batch_complete: Optional callback function called when each batch completes.
            save_multimedia_to_disk: Whether to save generated multimedia (images, audio, video) to disk.
                If False, multimedia is stored directly in the DataFrame (e.g., images as base64).
                Default is True.

        Returns:
            Path to the generated dataset directory.
        """
        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()
        self._write_builder_config()

        # Set media storage mode based on parameters
        if self._has_image_columns():
            mode = StorageMode.DISK if save_multimedia_to_disk else StorageMode.DATAFRAME
            self.artifact_storage.set_media_storage_mode(mode)

        generators = self._initialize_generators()
        start_time = time.perf_counter()
        group_id = uuid.uuid4().hex

        buffer_size = self._resource_provider.run_config.buffer_size
        self.batch_manager.start(num_records=num_records, buffer_size=buffer_size)
        for batch_idx in range(self.batch_manager.num_batches):
            logger.info(f"⏳ Processing batch {batch_idx + 1} of {self.batch_manager.num_batches}")
            self._run_batch(
                generators,
                batch_mode="batch",
                group_id=group_id,
                current_batch_number=batch_idx,
                on_batch_complete=on_batch_complete,
            )
        self.batch_manager.finish()
        self._processor_runner.run_after_generation(buffer_size)

        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return self.artifact_storage.final_dataset_path

    def build_preview(self, *, num_records: int) -> pd.DataFrame:
        self._run_model_health_check_if_needed()
        self._run_mcp_tool_check_if_needed()

        # Set media storage to DATAFRAME mode for preview - base64 stored directly in DataFrame
        if self._has_image_columns():
            self.artifact_storage.set_media_storage_mode(StorageMode.DATAFRAME)

        generators = self._initialize_generators()
        group_id = uuid.uuid4().hex
        start_time = time.perf_counter()
        self.batch_manager.start(num_records=num_records, buffer_size=num_records)
        self._run_batch(generators, batch_mode="preview", save_partial_results=False, group_id=group_id)
        dataset = self.batch_manager.get_current_batch(as_dataframe=True)
        self.batch_manager.reset()

        self._resource_provider.model_registry.log_model_usage(time.perf_counter() - start_time)

        return dataset

    def process_preview(self, dataset: pd.DataFrame) -> pd.DataFrame:
        df = self._processor_runner.run_post_batch(dataset.copy(), current_batch_number=None)
        return self._processor_runner.run_after_generation_on_df(df)

    def _has_image_columns(self) -> bool:
        """Check if config has any image generation columns."""
        return any(col.column_type == DataDesignerColumnType.IMAGE for col in self.single_column_configs)

    def _initialize_generators(self) -> list[ColumnGenerator]:
        return [
            self._registry.column_generators.get_for_config_type(type(config))(
                config=config, resource_provider=self._resource_provider
            )
            for config in self._column_configs
        ]

    def _write_builder_config(self) -> None:
        self.artifact_storage.mkdir_if_needed(self.artifact_storage.base_dataset_path)
        BuilderConfig(data_designer=self._data_designer_config).to_json(
            self.artifact_storage.base_dataset_path / SDG_CONFIG_FILENAME
        )

    def _run_batch(
        self,
        generators: list[ColumnGenerator],
        *,
        batch_mode: str,
        save_partial_results: bool = True,
        group_id: str,
        current_batch_number: int | None = None,
        on_batch_complete: Callable[[Path], None] | None = None,
    ) -> None:
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
        ran_pre_batch = False
        for generator in generators:
            generator.log_pre_generation()
            try:
                generation_strategy = generator.get_generation_strategy()
                if generator.can_generate_from_scratch and self.batch_manager.buffer_is_empty:
                    self._run_from_scratch_column_generator(generator)
                    # Run PRE_BATCH after seed generator, before other columns
                    if not ran_pre_batch:
                        self._processor_runner.run_pre_batch(self.batch_manager)
                        ran_pre_batch = True
                elif generation_strategy == GenerationStrategy.CELL_BY_CELL:
                    self._run_cell_by_cell_generator(generator)
                elif generation_strategy == GenerationStrategy.FULL_COLUMN:
                    self._run_full_column_generator(generator)
                else:
                    logger.error(f"❌ Unknown generation strategy: {generation_strategy}")
                    raise DatasetGenerationError(f"🛑 Unknown generation strategy: {generation_strategy}")
                if save_partial_results:
                    self.batch_manager.write()
            except Exception as e:
                column_error_str = (
                    f"columns {generator.config.column_names}"
                    if hasattr(generator.config, "column_names")
                    else f"column {generator.config.name!r}"
                )
                raise DatasetGenerationError(f"🛑 Failed to process {column_error_str}:\n{e}")

        try:
            usage_deltas = self._resource_provider.model_registry.get_usage_deltas(pre_batch_snapshot)
            self._emit_batch_inference_events(batch_mode, usage_deltas, group_id)
        except Exception:
            pass

        if current_batch_number is not None:
            df_batch = self.batch_manager.get_current_batch(as_dataframe=True)
            df_batch = self._processor_runner.run_post_batch(df_batch, current_batch_number=current_batch_number)
            self._write_processed_batch(df_batch)
            self.batch_manager.finish_batch(on_batch_complete)

    def _run_from_scratch_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate_from_scratch(self.batch_manager.num_records_batch)
        self.batch_manager.add_records(df.to_dict(orient="records"))

    def _run_cell_by_cell_generator(self, generator: ColumnGenerator) -> None:
        max_workers = self._resource_provider.run_config.non_inference_max_parallel_workers
        if isinstance(generator, ColumnGeneratorWithModel):
            max_workers = generator.inference_parameters.max_parallel_requests
        if DATA_DESIGNER_ASYNC_ENGINE:
            logger.info("⚡ Using async engine for concurrent execution")
            self._fan_out_with_async(generator, max_workers=max_workers)
        else:
            self._fan_out_with_threads(generator, max_workers=max_workers)

    def _run_full_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
        self.batch_manager.update_records(df.to_dict(orient="records"))

    def _run_model_health_check_if_needed(self) -> None:
        model_aliases: set[str] = set()
        for config in self.single_column_configs:
            if column_type_is_model_generated(config.column_type):
                model_aliases.add(config.model_alias)
            if isinstance(config, CustomColumnConfig) and config.model_aliases:
                model_aliases.update(config.model_aliases)

        if model_aliases:
            self._resource_provider.model_registry.run_health_check(list(model_aliases))

    def _run_mcp_tool_check_if_needed(self) -> None:
        tool_aliases = sorted(
            {config.tool_alias for config in self.llm_generated_column_configs if getattr(config, "tool_alias", None)}
        )
        if not tool_aliases:
            return
        if self._resource_provider.mcp_registry is None:
            raise DatasetGenerationError(f"Tool alias(es) {tool_aliases!r} specified but no MCPRegistry configured.")
        self._resource_provider.mcp_registry.run_health_check(tool_aliases)

    def _setup_fan_out(
        self, generator: ColumnGeneratorWithModelRegistry, max_workers: int
    ) -> tuple[ProgressTracker, dict[str, Any]]:
        if generator.get_generation_strategy() != GenerationStrategy.CELL_BY_CELL:
            raise DatasetGenerationError(
                f"Generator {generator.name} is not a {GenerationStrategy.CELL_BY_CELL} "
                "generator so concurrent fan-out is not supported."
            )

        progress_tracker = ProgressTracker(
            total_records=self.batch_manager.num_records_batch,
            label=f"{generator.config.column_type} column '{generator.config.name}'",
        )
        progress_tracker.log_start(max_workers)

        settings = self._resource_provider.run_config
        executor_kwargs: dict = {
            "column_name": generator.config.name,
            "result_callback": self._make_result_callback(progress_tracker),
            "error_callback": self._make_error_callback(progress_tracker),
            "shutdown_error_rate": settings.shutdown_error_rate,
            "shutdown_error_window": settings.shutdown_error_window,
            "disable_early_shutdown": settings.disable_early_shutdown,
        }

        return progress_tracker, executor_kwargs

    def _finalize_fan_out(self, progress_tracker: ProgressTracker) -> None:
        progress_tracker.log_final()
        if len(self._records_to_drop) > 0:
            self._cleanup_dropped_record_images(self._records_to_drop)
            self.batch_manager.drop_records(self._records_to_drop)
            self._records_to_drop.clear()

    def _fan_out_with_async(self, generator: ColumnGeneratorWithModelRegistry, max_workers: int) -> None:
        if getattr(generator.config, "tool_alias", None):
            logger.info("🛠️ Tool calling enabled")
        progress_tracker, executor_kwargs = self._setup_fan_out(generator, max_workers)
        executor = AsyncConcurrentExecutor(max_workers=max_workers, **executor_kwargs)
        work_items = [
            (generator.agenerate(record), {"index": i}) for i, record in self.batch_manager.iter_current_batch()
        ]
        executor.run(work_items)
        self._finalize_fan_out(progress_tracker)

    def _fan_out_with_threads(self, generator: ColumnGeneratorWithModelRegistry, max_workers: int) -> None:
        if getattr(generator.config, "tool_alias", None):
            logger.info("🛠️ Tool calling enabled")
        progress_tracker, executor_kwargs = self._setup_fan_out(generator, max_workers)
        with ConcurrentThreadExecutor(max_workers=max_workers, **executor_kwargs) as executor:
            for i, record in self.batch_manager.iter_current_batch():
                executor.submit(lambda record: generator.generate(record), record, context={"index": i})
        self._finalize_fan_out(progress_tracker)

    def _make_result_callback(self, progress_tracker: ProgressTracker) -> Callable[[dict], None]:
        def callback(result: dict, *, context: dict | None = None) -> None:
            self._worker_result_callback(result, context=context)
            progress_tracker.record_success()

        return callback

    def _make_error_callback(self, progress_tracker: ProgressTracker) -> Callable[[Exception], None]:
        def callback(exc: Exception, *, context: dict | None = None) -> None:
            self._worker_error_callback(exc, context=context)
            progress_tracker.record_failure()

        return callback

    def _write_processed_batch(self, dataframe: pd.DataFrame) -> None:
        self.batch_manager.update_records(dataframe.to_dict(orient="records"))
        self.batch_manager.write()

    def _validate_column_configs(self) -> None:
        if len(self._column_configs) == 0:
            raise DatasetGenerationError("🛑 No column configs provided.")

        if not self._registry.column_generators.get_for_config_type(
            type(self._column_configs[0])
        ).can_generate_from_scratch:
            raise DatasetGenerationError("🛑 The first column config must be a from-scratch column generator.")

    def _initialize_processors(self, processor_configs: list[ProcessorConfig]) -> list[Processor]:
        # Check columns marked for drop
        columns_to_drop = [config.name for config in self.single_column_configs if config.drop]

        processors: list[Processor] = []
        for config in processor_configs:
            processors.append(
                self._registry.processors.get_for_config_type(type(config))(
                    config=config,
                    resource_provider=self._resource_provider,
                )
            )

            # Manually included "drop columns" processor takes precedence
            if config.processor_type == ProcessorType.DROP_COLUMNS:
                for column in config.column_names:
                    if column in columns_to_drop:
                        columns_to_drop.remove(column)

        # If there are still columns marked for drop, add the "drop columns" processor to drop them
        if len(columns_to_drop) > 0:
            processors.append(
                DropColumnsProcessor(
                    config=DropColumnsProcessorConfig(
                        name="default_drop_columns_processor",
                        column_names=columns_to_drop,
                    ),
                    resource_provider=self._resource_provider,
                )
            )

        return processors

    def _cleanup_dropped_record_images(self, dropped_indices: set[int]) -> None:
        """Remove saved image files for records that will be dropped.

        When a record fails during generation, any images already saved to disk
        for that record in previous columns become dangling. This method deletes
        those files so they don't accumulate.
        """
        media_storage = self.artifact_storage.media_storage
        if not self._has_image_columns() or media_storage is None or media_storage.mode != StorageMode.DISK:
            return

        image_col_names = [
            col.name for col in self.single_column_configs if col.column_type == DataDesignerColumnType.IMAGE
        ]

        buffer = self.batch_manager.get_current_batch(as_dataframe=False)
        for idx in dropped_indices:
            if idx < 0 or idx >= len(buffer):
                continue
            for col_name in image_col_names:
                paths = buffer[idx].get(col_name, [])
                for path in [paths] if isinstance(paths, str) else paths:
                    media_storage.delete_image(path)

    def _worker_error_callback(self, exc: Exception, *, context: dict | None = None) -> None:
        """If a worker fails, we can handle the exception here."""
        logger.warning(
            f"⚠️ Generation for record at index {context['index']} failed. "
            f"Will omit this record from the dataset.\n{exc}"
        )
        self._records_to_drop.add(context["index"])

    def _worker_result_callback(self, result: dict, *, context: dict | None = None) -> None:
        self.batch_manager.update_record(context["index"], result)

    def _emit_batch_inference_events(
        self, batch_mode: str, usage_deltas: dict[str, ModelUsageStats], group_id: str
    ) -> None:
        if not usage_deltas:
            return

        events = [
            InferenceEvent(
                nemo_source=NemoSourceEnum.DATADESIGNER,
                task=batch_mode,
                task_status=TaskStatusEnum.SUCCESS,
                model=model_name,
                input_tokens=delta.token_usage.input_tokens,
                output_tokens=delta.token_usage.output_tokens,
            )
            for model_name, delta in usage_deltas.items()
        ]

        with TelemetryHandler(source_client_version=_CLIENT_VERSION, session_id=group_id) as telemetry_handler:
            for event in events:
                telemetry_handler.enqueue(event)
