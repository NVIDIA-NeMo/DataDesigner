# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import functools
import json
import logging
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from data_designer.config.columns import ColumnConfigT
from data_designer.engine.column_generators.generators.base import ColumnGenerator, GenerationStrategy
from data_designer.engine.column_generators.generators.llm_generators import WithLLMGeneration
from data_designer.engine.dataset_builders.artifact_storage import ArtifactStorage, BatchStage
from data_designer.engine.dataset_builders.errors import DatasetGenerationError
from data_designer.engine.dataset_builders.multi_column_configs import (
    DatasetBuilderColumnConfigT,
    MultiColumnConfig,
)
from data_designer.engine.dataset_builders.utils.concurrency import (
    MAX_CONCURRENCY_PER_NON_LLM_GENERATOR,
    ConcurrentThreadExecutor,
)
from data_designer.engine.dataset_builders.utils.dataset_batch_manager import (
    DatasetBatchManager,
)
from data_designer.engine.processing.processors.configs import DropColumnsProcessorConfig
from data_designer.engine.processing.processors.drop_columns import DropColumnsProcessor
from data_designer.engine.registry.data_designer_registry import DataDesignerRegistry
from data_designer.engine.resources.resource_provider import ResourceProvider

logger = logging.getLogger(__name__)


class ColumnWiseDatasetBuilder:
    def __init__(
        self,
        column_configs: list[DatasetBuilderColumnConfigT],
        resource_provider: ResourceProvider,
        registry: DataDesignerRegistry | None = None,
    ):
        self.batch_manager = DatasetBatchManager(resource_provider.artifact_storage)
        self._resource_provider = resource_provider
        self._records_to_drop: set[int] = set()
        self._registry = registry or DataDesignerRegistry()
        self._column_configs = column_configs
        self._validate_column_configs()

    @property
    def artifact_storage(self) -> ArtifactStorage:
        return self._resource_provider.artifact_storage

    @functools.cached_property
    def single_column_configs(self) -> list[ColumnConfigT]:
        configs = []
        for config in self._column_configs:
            if isinstance(config, ColumnConfigT):
                configs.append(config)
            elif isinstance(config, MultiColumnConfig):
                configs.extend(config.columns)
        return configs

    def build(
        self,
        *,
        num_records: int,
        buffer_size: int,
        on_batch_complete: Callable[[Path], None] | None = None,
    ) -> Path:
        self._write_configs()
        self._run_model_health_check_if_needed()

        generators = self._initialize_generators()
        start_time = time.perf_counter()

        self.batch_manager.start(num_records=num_records, buffer_size=buffer_size)
        for batch_idx in range(1, self.batch_manager.num_batches + 1):
            logger.info(f"⏳ Processing batch {batch_idx} of {self.batch_manager.num_batches}")
            self._run_batch(generators)
            df_batch = self.batch_manager.get_current_batch(as_dataframe=True)
            self._write_processed_batch(self.drop_columns_if_needed(df_batch, save_dropped_columns=True))
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

        start_time = time.perf_counter()
        self.batch_manager.start(num_records=num_records, buffer_size=num_records)
        self._run_batch(generators, save_partial_results=False)
        dataset = self.batch_manager.get_current_batch(as_dataframe=True)
        self.batch_manager.reset()

        model_usage_stats = self._resource_provider.model_registry.get_model_usage_stats(
            time.perf_counter() - start_time
        )
        logger.info(f"📊 Model usage summary:\n{json.dumps(model_usage_stats, indent=4)}")

        return dataset

    def drop_columns_if_needed(self, dataframe: pd.DataFrame, *, save_dropped_columns: bool = False) -> pd.DataFrame:
        if len(columns_to_drop := [config.name for config in self.single_column_configs if config.drop]) == 0:
            return dataframe
        try:
            dropped_column_parquet_file_name = (
                None
                if not save_dropped_columns
                else self.artifact_storage.create_batch_file_path(
                    batch_number=self.batch_manager.get_current_batch_number(),
                    batch_stage=BatchStage.DROPPED_COLUMNS,
                ).name
            )
            df = DropColumnsProcessor(
                config=DropColumnsProcessorConfig(
                    column_names=columns_to_drop,
                    dropped_column_parquet_file_name=dropped_column_parquet_file_name,
                ),
                resource_provider=self._resource_provider,
            ).process(dataframe)
            return df
        except Exception as e:
            raise DatasetGenerationError(f"🛑 Failed to drop columns {columns_to_drop}: {e}")

    def _initialize_generators(self) -> list[ColumnGenerator]:
        return [
            self._registry.column_generators.get_for_config_type(type(config))(
                config=config, resource_provider=self._resource_provider
            )
            for config in self._column_configs
        ]

    def _run_batch(self, generators: list[ColumnGenerator], *, save_partial_results: bool = True) -> None:
        for generator in generators:
            generator.log_pre_generation()
            try:
                if generator.can_generate_from_scratch and self.batch_manager.buffer_is_empty:
                    self._run_from_scratch_column_generator(generator)
                elif generator.generation_strategy == GenerationStrategy.CELL_BY_CELL:
                    self._run_cell_by_cell_generator(generator)
                elif generator.generation_strategy == GenerationStrategy.FULL_COLUMN:
                    self._run_full_column_generator(generator)
                else:
                    logger.error(f"❌ Unknown generation strategy: {generator.generation_strategy}")
                    raise DatasetGenerationError(f"🛑 Unknown generation strategy: {generator.generation_strategy}")
                if save_partial_results:
                    self.batch_manager.write()
            except Exception as e:
                column_error_str = (
                    f"columns {generator.config.column_names}"
                    if hasattr(generator.config, "column_names")
                    else f"column {generator.config.name!r}"
                )
                raise DatasetGenerationError(f"🛑 Failed to process {column_error_str}:\n{e}")

    def _run_from_scratch_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate_from_scratch(self.batch_manager.num_records_batch)
        self.batch_manager.add_records(df.to_dict(orient="records"))

    def _run_cell_by_cell_generator(self, generator: ColumnGenerator) -> None:
        max_workers = MAX_CONCURRENCY_PER_NON_LLM_GENERATOR
        if isinstance(generator, WithLLMGeneration):
            max_workers = generator.inference_parameters.max_parallel_requests
        self._fan_out_with_threads(generator, max_workers=max_workers)

    def _run_full_column_generator(self, generator: ColumnGenerator) -> None:
        df = generator.generate(self.batch_manager.get_current_batch(as_dataframe=True))
        self.batch_manager.update_records(df.to_dict(orient="records"))

    def _run_model_health_check_if_needed(self) -> bool:
        if any(config.column_type.has_prompt_templates for config in self.single_column_configs):
            self._resource_provider.model_registry.run_health_check()

    def _fan_out_with_threads(self, generator: WithLLMGeneration, max_workers: int) -> None:
        if generator.generation_strategy != GenerationStrategy.CELL_BY_CELL:
            raise DatasetGenerationError(
                f"Generator {generator.metadata().name} is not a {GenerationStrategy.CELL_BY_CELL} "
                "generator so concurrency through threads is not supported."
            )

        logger.info(
            f"🐙 Processing {generator.config.column_type.value} column '{generator.config.name}' "
            f"with {max_workers} concurrent workers"
        )
        with ConcurrentThreadExecutor(
            max_workers=max_workers,
            column_name=generator.config.name,
            result_callback=self._worker_result_callback,
            error_callback=self._worker_error_callback,
        ) as executor:
            for i, record in self.batch_manager.iter_current_batch():
                executor.submit(lambda record: generator.generate(record), record, context={"index": i})

        if len(self._records_to_drop) > 0:
            self.batch_manager.drop_records(self._records_to_drop)
            self._records_to_drop.clear()

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

    def _worker_error_callback(self, exc: Exception, *, context: dict | None = None) -> None:
        """If a worker fails, we can handle the exception here."""
        logger.warning(
            f"⚠️ Generation for record at index {context['index']} failed. "
            f"Will omit this record from the dataset.\n{exc}"
        )
        self._records_to_drop.add(context["index"])

    def _worker_result_callback(self, result: dict, *, context: dict | None = None) -> None:
        self.batch_manager.update_record(context["index"], result)

    def _write_configs(self) -> None:
        self.artifact_storage.write_configs(
            json_file_name="column_configs.json",
            configs=self._column_configs,
        )
        self.artifact_storage.write_configs(
            json_file_name="model_configs.json",
            configs=self._resource_provider.model_registry.model_configs.values(),
        )
