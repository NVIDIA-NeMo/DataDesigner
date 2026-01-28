# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import functools
import importlib.metadata
import json
import logging
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from data_designer.config.column_types import ColumnConfigT
from data_designer.config.config_builder import BuilderConfig
from data_designer.config.data_designer_config import DataDesignerConfig
from data_designer.config.dataset_builders import BuildStage
from data_designer.config.processors import (
    DropColumnsProcessorConfig,
    ProcessorConfig,
    ProcessorType,
)
from data_designer.engine.column_generators.generators.base import (
    ColumnGenerator,
    ColumnGeneratorWithModel,
    GenerationStrategy,
)
from data_designer.engine.column_generators.utils.generator_classification import column_type_is_model_generated
from data_designer.engine.compiler import compile_data_designer_config
from data_designer.engine.dataset_builders.artifact_storage import SDG_CONFIG_FILENAME, ArtifactStorage
from data_designer.engine.dataset_builders.errors import DatasetGenerationError, DatasetProcessingError
from data_designer.engine.dataset_builders.multi_column_configs import MultiColumnConfig
from data_designer.engine.dataset_builders.utils.concurrency import ConcurrentThreadExecutor
from data_designer.engine.dataset_builders.utils.config_compiler import compile_dataset_builder_column_configs
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import DatasetBatchManager
from data_designer.engine.dataset_builders.utils.progress_tracker import ProgressTracker
from data_designer.engine.models.telemetry import InferenceEvent, NemoSourceEnum, TaskStatusEnum, TelemetryHandler
from data_designer.engine.processing.processors.base import Processor
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider
from data_designer.lazy_heavy_imports import pd

if TYPE_CHECKING:
    import pandas as pd

    from data_designer.engine.column_generators.generators.base import ColumnGeneratorWithModelRegistry
    from data_designer.engine.models.usage import ModelUsageStats

logger = logging.getLogger(__name__)

_CLIENT_VERSION: str = importlib.metadata.version("data-designer-engine")


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
        self._processors: dict[BuildStage, list[Processor]] = self._initialize_processors(
            self._data_designer_config.processors or []
        )
        self._validate_column_configs()

    @property
    def artifact_storage(self) -> ArtifactStorage:
        return self._resource_provider.artifact_storage

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
    ) -> Path:
        self._run_model_health_check_if_needed()
        self._write_builder_config()
        generators = self._initialize_generators()
        start_time = time.perf_counter()
        group_id = uuid.uuid4().hex

        buffer_size = self._resource_provider.run_config.buffer_size
        self.batch_manager.start(num_records=num_records, buffer_size=buffer_size)
        for batch_idx in range(self.batch_manager.num_batches):
            logger.info(f"⏳ Processing batch {batch_idx + 1} of {self.batch_manager.num_batches}")
            self._run_batch(generators, batch_mode="batch", group_id=group_id)
            df_batch = self._run_processors(
                stage=BuildStage.POST_BATCH,
                dataframe=self.batch_manager.get_current_batch(as_dataframe=True),
                current_batch_number=batch_idx,
            )
            self._write_processed_batch(df_batch)
            self.batch_manager.finish_batch(on_batch_complete)
        self.batch_manager.finish()

        model_usage_stats = self._resource_provider.model_registry.get_model_usage_stats(
            time.perf_counter() - start_time
        )
        logger.info(f"📊 Model usage summary:\n{json.dumps(model_usage_stats, indent=4)}")

        return self.artifact_storage.final_dataset_path

    def build_preview(self, *, num_records: int) -> pd.DataFrame:
        self._run_model_health_check_if_needed()

        generators = self._initialize_generators()
        group_id = uuid.uuid4().hex
        start_time = time.perf_counter()
        self.batch_manager.start(num_records=num_records, buffer_size=num_records)
        self._run_batch(generators, batch_mode="preview", save_partial_results=False, group_id=group_id)
        dataset = self.batch_manager.get_current_batch(as_dataframe=True)
        self.batch_manager.reset()

        model_usage_stats = self._resource_provider.model_registry.get_model_usage_stats(
            time.perf_counter() - start_time
        )
        logger.info(f"📊 Model usage summary:\n{json.dumps(model_usage_stats, indent=4)}")

        return dataset

    def process_preview(self, dataset: pd.DataFrame) -> pd.DataFrame:
        return self._run_processors(
            stage=BuildStage.POST_BATCH,
            dataframe=dataset.copy(),
            current_batch_number=None,  # preview mode does not have a batch number
        )

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
        self, generators: list[ColumnGenerator], *, batch_mode: str, save_partial_results: bool = True, group_id: str
    ) -> None:
        pre_batch_snapshot = self._resource_provider.model_registry.get_model_usage_snapshot()
        for generator in generators:
            generator.log_pre_generation()
            try:
                generation_strategy = generator.get_generation_strategy()
                if generator.can_generate_from_scratch and self.batch_manager.buffer_is_empty:
                    self._run_from_scratch_column_generator(generator)
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

    def _run_from_scratch_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate_from_scratch(self.batch_manager.num_records_batch)
        self.batch_manager.add_records(df.to_dict(orient="records"))

    def _run_cell_by_cell_generator(self, generator: ColumnGenerator) -> None:
        max_workers = self._resource_provider.run_config.non_inference_max_parallel_workers
        if isinstance(generator, ColumnGeneratorWithModel):
            max_workers = generator.inference_parameters.max_parallel_requests
        self._fan_out_with_threads(generator, max_workers=max_workers)

    def _run_full_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
        self.batch_manager.update_records(df.to_dict(orient="records"))

    def _run_model_health_check_if_needed(self) -> bool:
        if any(column_type_is_model_generated(config.column_type) for config in self.single_column_configs):
            self._resource_provider.model_registry.run_health_check(
                list(set(config.model_alias for config in self.llm_generated_column_configs))
            )

    def _fan_out_with_threads(self, generator: ColumnGeneratorWithModelRegistry, max_workers: int) -> None:
        if generator.get_generation_strategy() != GenerationStrategy.CELL_BY_CELL:
            raise DatasetGenerationError(
                f"Generator {generator.name} is not a {GenerationStrategy.CELL_BY_CELL} "
                "generator so concurrency through threads is not supported."
            )

        progress_tracker = ProgressTracker(
            total_records=self.batch_manager.num_records_batch,
            label=f"{generator.config.column_type} column '{generator.config.name}'",
        )
        progress_tracker.log_start(max_workers)

        settings = self._resource_provider.run_config
        with ConcurrentThreadExecutor(
            max_workers=max_workers,
            column_name=generator.config.name,
            result_callback=self._make_result_callback(progress_tracker),
            error_callback=self._make_error_callback(progress_tracker),
            shutdown_error_rate=settings.shutdown_error_rate,
            shutdown_error_window=settings.shutdown_error_window,
            disable_early_shutdown=settings.disable_early_shutdown,
        ) as executor:
            for i, record in self.batch_manager.iter_current_batch():
                executor.submit(lambda record: generator.generate(record), record, context={"index": i})

        progress_tracker.log_final()

        if len(self._records_to_drop) > 0:
            self.batch_manager.drop_records(self._records_to_drop)
            self._records_to_drop.clear()

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

    def _initialize_processors(self, processor_configs: list[ProcessorConfig]) -> dict[BuildStage, list[Processor]]:
        # Check columns marked for drop
        columns_to_drop = [config.name for config in self.single_column_configs if config.drop]

        processors: dict[BuildStage, list[Processor]] = {stage: [] for stage in BuildStage}
        for config in processor_configs:
            processors[config.build_stage].append(
                self._registry.processors.get_for_config_type(type(config))(
                    config=config,
                    resource_provider=self._resource_provider,
                )
            )

            # Manually included "drop columns" processor takes precedence (can e.g., pick stages other than post-batch)
            if config.processor_type == ProcessorType.DROP_COLUMNS:
                for column in config.column_names:
                    if column in columns_to_drop:
                        columns_to_drop.remove(column)

        # If there are still columns marked for drop, add the "drop columns" processor to drop them
        if len(columns_to_drop) > 0:
            processors[BuildStage.POST_BATCH].append(  # as post-batch by default
                DropColumnsProcessor(
                    config=DropColumnsProcessorConfig(
                        name="default_drop_columns_processor",
                        column_names=columns_to_drop,
                        build_stage=BuildStage.POST_BATCH,
                    ),
                    resource_provider=self._resource_provider,
                )
            )

        return processors

    def _run_processors(
        self, stage: BuildStage, dataframe: pd.DataFrame, current_batch_number: int | None = None
    ) -> pd.DataFrame:
        for processor in self._processors[stage]:
            try:
                dataframe = processor.process(dataframe, current_batch_number=current_batch_number)
            except Exception as e:
                raise DatasetProcessingError(
                    f"🛑 Failed to process dataset with processor {processor.name} in stage {stage}: {e}"
                ) from e
        return dataframe

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
